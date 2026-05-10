# src/worker.py
import argparse
import logging
import time

# os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from config.manager import load_config
from src.metrics.energy import EnergyMeasurement
from src.queue.redis_client import RedisClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self.tokenizer = tokenizer
        self.stop_word_ids = [
            self.tokenizer.encode(word, add_special_tokens=False) for word in stop_words
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_word_ids:
            if len(stop_ids) == 0:
                continue

            if (
                input_ids.shape[1] >= len(stop_ids)
                and input_ids[0][-len(stop_ids) :].tolist() == stop_ids
            ):
                return True
        return False


class InferenceService:
    """Worker that pulls tasks from Redis and runs inference"""

    def __init__(
        self,
        redis_client,
        general_configs,
        model_configs,
        device_id=0,
        energy_method="pynvml",
        tier="cloud",
        dedicated_queue=None
    ):
        self.redis_client = redis_client
        self.general_configs = general_configs
        self.model_configs = model_configs
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.energy_measurement = EnergyMeasurement.create(
            method=energy_method, device_id=device_id
        )
        self.loaded_model = None
        self.loaded_model_id = None
        self.tokenizer = None
        self.tier = tier
        self.queue_name = dedicated_queue or f"llm_inference_tasks_{self.tier}"

    def _load_model(self, model_id):
        """Load model if not already loaded"""
        if self.loaded_model_id == model_id:
            return self.loaded_model, self.tokenizer
        if self.loaded_model is not None:
            logger.info(f"Unloading model {self.loaded_model_id}")
            self.loaded_model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

        logger.info(f"Loading model {model_id}")
        model_config = self.model_configs[model_id]
        model_name = model_config["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.general_configs.get("cache_dir"),
            token=self.general_configs.get("hf_token"),
        )
        self.loaded_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir=self.general_configs.get("cache_dir"),
            attn_implementation="eager",
        )
        self.loaded_model_id = model_id
        self._warm_up_model(self.loaded_model, self.tokenizer)

        return self.loaded_model, self.tokenizer

    def run_inference(self, query_text, model_id, generation_parameters=None):
        """Run inference on a query"""
        model, tokenizer = self._load_model(model_id)
        inputs, input_tokens, gen_params = self._prepare_generation_params(
            query_text, tokenizer, generation_parameters
        )
        self.energy_measurement.start()
        start_time = time.time()

        logger.debug(
            f"Generating response for input starting with: {query_text[:100]}..."
        )
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, **gen_params)
        latency = time.time() - start_time
        energy = self.energy_measurement.stop()
        output_tokens = outputs.shape[1] - input_tokens
        output_text = tokenizer.decode(
            outputs[0, input_tokens:], skip_special_tokens=True
        )

        return {
            "response": output_text,
            "latency": latency,
            "metrics": {
                "latency": latency,
                "energy_consumption": energy,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    def _warm_up_model(self, model, tokenizer):
        """Run a warm-up inference to initialize model compilation and caches"""
        logger.info(f"Warming up model {self.loaded_model_id}")
        try:
            warm_up_text = "Hello, how are you today?"
            inputs = tokenizer(warm_up_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                _ = model.generate(inputs.input_ids, max_new_tokens=20, do_sample=False)
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _prepare_generation_params(
        self, query_text, tokenizer, generation_parameters=None
    ):
        """Prepare generation parameters and inputs"""

        inputs = tokenizer(
            query_text, return_tensors="pt", truncation=True, max_length=8192
        ).to(self.device)
        input_tokens = inputs.input_ids.shape[1]
        gen_params = {
            "max_new_tokens": 256,
            "do_sample": True,
        }
        if generation_parameters:
            if "stop_sequence" in generation_parameters:
                stop_sequences = generation_parameters.pop("stop_sequence")
                if isinstance(stop_sequences, str):
                    try:
                        import ast

                        stop_sequences = ast.literal_eval(stop_sequences)
                    except (ValueError, SyntaxError):
                        stop_sequences = [stop_sequences]
                if not isinstance(stop_sequences, list):
                    stop_sequences = [stop_sequences]
                if stop_sequences:
                    stopping_criteria = StoppingCriteriaList()
                    stopping_criteria.append(
                        KeywordsStoppingCriteria(tokenizer, stop_sequences)
                    )
                    gen_params["stopping_criteria"] = stopping_criteria
            for key, value in generation_parameters.items():
                if key in ["max_new_tokens", "max_length"]:
                    gen_params["max_new_tokens"] = int(value)
                elif key in ["temperature", "top_p", "top_k", "num_beams"]:
                    gen_params[key] = value

        return inputs, input_tokens, gen_params

    def run(self):
        """Main worker loop"""
        logger.info(f"Worker started on {self.device} listening to queue: {self.queue_name}")

        while True:
            task = self.redis_client.get_next_task(queue_name=self.queue_name, timeout=1)
            if not task:
                time.sleep(0.1)
                continue

            logger.info(
                f"Processing task {task['id']} with model {task['selected_model']}"
            )

            try:
                result = self.run_inference(
                    task["query_text"],
                    task["selected_model"],
                    task.get("generation_parameters"),
                )
                self.redis_client.set_result(
                    task["id"],
                    {
                        **result,
                        "status": "completed",
                        "selected_model": task["selected_model"],
                        "extraction_method": task.get("extraction_method"),
                        "evaluation_metric": task.get("evaluation_metric"),
                    },
                )

                logger.info(f"Completed task {task['id']}")
            except Exception as e:
                logger.error(f"Error processing task {task['id']}: {e}")
                self.redis_client.set_result(
                    task["id"],
                    {
                        "error": str(e),
                        "status": "failed",
                        "selected_model": task["selected_model"],
                    },
                )


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Worker")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument(
        "--redis-password", default=None, help="Redis password (if required)"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--energy-method",
        default="zeus",
        choices=["pynvml", "zeus"],
        help="Energy measurement method",
    )
    parser.add_argument("--tier", default="cloud", choices=["cloud", "edge"], help="Is this an edge or cloud worker?")
    parser.add_argument(
        "--dedicated-queue",
        default=None,
        help="Override the default tier queue with a specific queue name (e.g. llm_tasks_node_3)",
    )
    args = parser.parse_args()
    redis_client = RedisClient(
        host=args.redis_host, port=args.redis_port, password=args.redis_password
    )

    model_configs = load_config("models")
    general_configs = load_config("general")
    worker = InferenceService(
        redis_client,
        general_configs,
        model_configs,
        device_id=args.device,
        energy_method=args.energy_method,
        tier=args.tier,
        dedicated_queue=args.dedicated_queue,
    )
    worker.run()


if __name__ == "__main__":
    main()

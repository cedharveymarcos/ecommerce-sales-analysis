# Chat with MLX

![MLX Chat UI](https://img.shields.io/badge/Apple%20Silicon-MLX-blue?style=for-the-badge&logo=apple)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Platform](https://img.shields.io/badge/Platform-macOS%2013.5+-lightgrey?style=for-the-badge&logo=apple)

> [!IMPORTANT]
> **Active Development Notice**: This project is under active development. Stay updated with the latest features:


**Run LLM on your Mac!** An all-in-one LLMs chat Web UI based on the MLX framework, designed for Apple Silicon.


chat-with-mlx provides a similar and more modern experience, and offers more features.

You can **upload files**, or even **images** to chat when using the vision model.

You can also use the **reasoning model** and see the model's reasoning process.

Also supports online LLM using OpenAI API compatible

If this helps you, I'd be happy if you could give me a star, thank you. ✨

## Quick Start

Get up and running in 30 seconds:

```bash
# Create virtual environment
python -m venv chat-with-mlx
cd chat-with-mlx
source bin/activate

That's it! Open http://127.0.0.1:7860 in your browser and start chatting! 🎉

## Model Support

* Qwen3
* Phi-4-reasoning
* DeepSeek-R1-0528
* Gemma3
* and so on...

## Roadmap

### Key Features

* [x] Chat
* [x] Completion
* [x] Model Management
* [x] RAG
* [ ] Function Call
* [ ] MCP 

### Others

* [x] Upload file to chat（Support `PDF, Word, Excel, PPT` and some plain text file like `.txt, .csv, .md`）
* [x] Upload picture to chat (Tested on the `Phi-3.5-vision-instruct` and `Qwen2.5-VL-7B-Instruct`)
* [x] See the model's reasoning process (Tested on `Qwen3` and `Phi-4-reasoning`)
* [ ] and so on...

## How to use

### Installation

1. Install using pip:
   ```bash
   python -m venv chat-with-mlx
   cd chat-with-mlx
   . ./bin/activate
   ```

### Run

2. Start the server:
   ```bash
   chat-with-mlx
   ```

- `--port`: The port on which the server will run (default is `7860`).
- `--share`: If specified, the server will be shared publicly.

### Use

3. Use in browser: By default, a page will open, http://127.0.0.1:7860, where you can chat.

### Model Configuration

You no longer need to add models by manually editing configuration files, you only need to use the "Model Management" page to add your models.

You can add various models from [mlx-community](https://huggingface.co/mlx-community). Models will be automatically downloaded from HuggingFace.

For the following configuration files, the model files will be stored in `models/models/Ministral-8B-Instruct-2410-4bit`.

**Ministral-8B-Instruct-2410-4bit.json**

```json
{
  "original_repo": "mistralai/Ministral-8B-Instruct-2410",
  "mlx_repo": "mlx-community/Ministral-8B-Instruct-2410-4bit",
  "model_name": "Ministral-8B-Instruct-2410-4bit",
  "quantize": "4bit",
  "default_language": "multi",
  "system_prompt": "",
  "multimodal_ability": []
}
```

- `original_repo`: The original repository where the model can be found.
- `mlx_repo`: The repository in the MLX community.
- `model_name`: The name of the model.
- `quantize`: The quantization format of the model (e.g., `4bit`).
- `default_language`: Default language setting (e.g., `multi` for multilingual support).
- `system_prompt`: The system prompt of the model.
- `multimodal_ability`: The multimodal capabilities of the model.

## License

This project is licensed under the MIT License.

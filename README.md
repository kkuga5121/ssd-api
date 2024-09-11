# SSD API
3. Install the dependencies using Poetry:

```bash
poetry install
```

## Usage

1. Start the FastAPI server:

```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0
```

2. Open your web browser and go to `http://localhost:8000` to access the text-to-speech window.

3. Enter the desired text in the input field and click the "Generate Speech" button.

4. The generated speech output will be played in the browser.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

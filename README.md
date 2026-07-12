# MDR / MDCG Regulatory RAG

## Streamlit Demo

This browser UI lets a professor test the MDR/MDCG RAG as a normal user.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your Gemini key to `.env`:

```bash
GEMINI_API_KEY=your_key_here
STREAMLIT_APP_PASSWORD=choose_a_password_here
```

3. Make sure indexes already exist. If needed, rebuild them:

```bash
python src/build_index.py
```

4. Run the UI:

```bash
streamlit run src/ui/streamlit_app.py
```

The app logs requests to `logs/usage.log`. API keys are read only from `.env` or environment variables and are not shown in the UI.

## Sharing With ngrok

Keep the Streamlit app running locally:

```bash
streamlit run src/ui/streamlit_app.py
```

In another terminal, expose it with ngrok:

```bash
ngrok http 8501
```

Send the public `https://...ngrok-free.app` URL to the tester together with the `STREAMLIT_APP_PASSWORD`. Anyone with the URL and password can use your Gemini quota while the tunnel is running.

# Alpha-Beta Web App (Shareable)

This is a Streamlit web version of the alpha-beta symmetric point design tool.

## Files
- `wind_tunnel_alpha_beta_web_app.py`: web app
- `requirements.txt`: Python dependencies

## Local Run

From this folder:

```bash
cd /Users/paras.singh/Documents/EV3_Droid/Wind_tunnel_testing/Codes/WTT3/test
python3 -m venv .venv_web
source .venv_web/bin/activate
pip install -r requirements.txt
streamlit run wind_tunnel_alpha_beta_web_app.py
```

## Share With Anyone (Streamlit Cloud)

1. Push this folder to a GitHub repo.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and point it to:
   - repo branch
   - file: `wind_tunnel_alpha_beta_web_app.py`
4. Deploy.
5. Share the public URL.

Users can use the tool in browser and click **Download Active Points CSV** to save on their own machine.

## Notes
- Manual point input format is `(beta,alpha)`.
- Plot selection uses box/lasso; then click **Remove Selected** or **Restore Selected**.
- Symmetry is enforced for remove/restore and point-add actions.

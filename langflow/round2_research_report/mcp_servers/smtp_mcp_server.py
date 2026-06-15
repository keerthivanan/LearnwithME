"""
Email (SMTP) MCP Server — sends email via Gmail
================================================
Exposes a `send_email` tool over the Model Context Protocol (MCP).
The Email Agent in Langflow connects to this via the MCP Tools component.

SETUP (one time):
  1. Turn on 2-Step Verification on your Google account.
  2. Create an "App Password": Google Account -> Security -> App passwords.
  3. Use that 16-char app password (NOT your normal Gmail password).

RUN IT (with uv — auto-installs deps). Needs two env vars:
     command: uv run --with mcp python smtp_mcp_server.py
     env:     GMAIL_ADDRESS=you@gmail.com
              GMAIL_APP_PASSWORD=your16charapppassword
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("smtp-email")


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Provide the recipient address (to), a subject line, and the
    full body text. Use this to deliver a finished report to the recipient.
    Returns a confirmation string, or an error message if sending fails.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return "CONFIG_ERROR: GMAIL_ADDRESS / GMAIL_APP_PASSWORD env vars are not set."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(sender, password)
            server.sendmail(sender, to, msg.as_string())
        return f"SUCCESS: email sent to {to} with subject '{subject}'."
    except smtplib.SMTPAuthenticationError:
        return "AUTH_ERROR: Gmail rejected the login. Check the app password and that 2FA is on."
    except Exception as e:
        return f"SEND_ERROR: could not send the email ({e})."


if __name__ == "__main__":
    mcp.run()

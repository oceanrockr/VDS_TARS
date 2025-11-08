"""
T.A.R.S. WebSocket Client Example
Demonstrates how to connect to the T.A.R.S. WebSocket gateway
"""

import asyncio
import json
import websockets
import httpx
from datetime import datetime


class TARSClient:
    """Simple T.A.R.S. WebSocket client"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.token = None
        self.websocket = None

    async def authenticate(self, client_id: str, device_name: str = None):
        """
        Authenticate and obtain JWT token

        Args:
            client_id: Unique client identifier
            device_name: Optional device name
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/auth/token",
                json={
                    "client_id": client_id,
                    "device_name": device_name or "example-client",
                    "device_type": "python",
                },
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                print(f"✓ Authenticated as {client_id}")
                print(f"  Token expires in {data['expires_in'] / 3600:.1f} hours")
                return True
            else:
                print(f"✗ Authentication failed: {response.status_code}")
                return False

    async def connect(self):
        """Connect to WebSocket endpoint"""
        if not self.token:
            raise ValueError("Must authenticate before connecting")

        ws_endpoint = f"{self.ws_url}/ws/chat?token={self.token}"

        try:
            self.websocket = await websockets.connect(ws_endpoint)
            print(f"✓ WebSocket connected")

            # Receive connection acknowledgment
            ack = await self.websocket.recv()
            ack_data = json.loads(ack)

            if ack_data["type"] == "connection_ack":
                print(f"  Session ID: {ack_data['session_id']}")
                return True
            else:
                print(f"✗ Unexpected message: {ack_data}")
                return False

        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    async def send_message(self, content: str, conversation_id: str = None):
        """
        Send a chat message

        Args:
            content: Message content
            conversation_id: Optional conversation ID for threading
        """
        if not self.websocket:
            raise ValueError("Must connect before sending messages")

        message = {
            "type": "chat",
            "content": content,
            "conversation_id": conversation_id or f"conv-{datetime.now().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.websocket.send(json.dumps(message))
        print(f"\n→ Sent: {content}")

    async def receive_stream(self):
        """
        Receive streaming response

        Yields tokens as they arrive
        """
        full_response = []

        try:
            while True:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=30.0
                )

                data = json.loads(message)
                msg_type = data["type"]

                if msg_type == "token":
                    token = data["token"]
                    full_response.append(token)
                    print(token, end="", flush=True)

                elif msg_type == "complete":
                    print(f"\n\n✓ Complete ({data['total_tokens']} tokens in {data['latency_ms']:.0f}ms)")
                    return "".join(full_response)

                elif msg_type == "error":
                    print(f"\n✗ Error: {data['error']}")
                    return None

                elif msg_type == "ping":
                    # Respond to ping
                    pong = {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await self.websocket.send(json.dumps(pong))

                elif msg_type == "system":
                    print(f"\n⚠ System: {data['message']}")

        except asyncio.TimeoutError:
            print(f"\n✗ Timeout waiting for response")
            return None

    async def chat(self, message: str):
        """
        Send a message and receive the complete response

        Args:
            message: Chat message

        Returns:
            Complete response text
        """
        await self.send_message(message)
        return await self.receive_stream()

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            print("\n✓ Connection closed")


async def main():
    """Example usage"""
    print("=== T.A.R.S. WebSocket Client Example ===\n")

    # Create client
    client = TARSClient(base_url="http://localhost:8000")

    # Authenticate
    authenticated = await client.authenticate(
        client_id="example-client-001",
        device_name="Python Example"
    )

    if not authenticated:
        return

    # Connect to WebSocket
    connected = await client.connect()

    if not connected:
        return

    # Chat examples
    try:
        # Example 1: Simple question
        print("\n" + "=" * 60)
        print("Example 1: Simple Question")
        print("=" * 60)

        response = await client.chat("What is the capital of France?")

        # Example 2: Follow-up question
        print("\n" + "=" * 60)
        print("Example 2: Follow-up")
        print("=" * 60)

        response = await client.chat("What is its population?")

        # Example 3: Technical question
        print("\n" + "=" * 60)
        print("Example 3: Technical Question")
        print("=" * 60)

        response = await client.chat("Explain what a WebSocket is in one sentence.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Close connection
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

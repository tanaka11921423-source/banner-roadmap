"""
ai_provider_router.py
ナイトワーク系バナー生成 — AIプロバイダー切り替えルーター

対応プロバイダー:
  - Adobe Firefly (Firefly Image 3 API)
  - Google Gemini 2.0 Flash (Imagen 3経由)
  - OpenAI ChatGPT / DALL-E 3

使い方:
  router = AIProviderRouter(provider="firefly")  # or "gemini" / "chatgpt"
  result = router.generate_image(prompt="高級クラブのバナー背景、深紅と黒、ラグジュアリー")
  # result.image_bytes → Pillowへ渡してバナー合成
"""

import os
import base64
import requests
from dataclasses import dataclass
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未インストール時はスキップ（手動で環境変数を設定）

ProviderType = Literal["firefly", "gemini", "chatgpt"]


@dataclass
class GenerationResult:
    """全プロバイダー共通の戻り値"""
    image_bytes: bytes        # PNG/JPGのバイト列
    provider: str             # 使用したプロバイダー名
    prompt_used: str          # 実際に送ったプロンプト
    width: int
    height: int


class AIProviderRouter:
    """
    プロバイダーを provider= で切り替えるだけで
    同じインターフェースで画像生成できるルーター
    """

    # APIキーは環境変数から読む（コードに直書きしない）
    FIREFLY_CLIENT_ID     = os.environ.get("ADOBE_CLIENT_ID", "")
    FIREFLY_CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET", "")
    GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "")
    OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")

    def __init__(self, provider: ProviderType = "firefly"):
        self.provider = provider

    # ------------------------------------------------------------------ #
    # 公開メソッド：UIから呼ぶのはここだけ
    # ------------------------------------------------------------------ #
    def generate_image(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1536,
        style: str = "luxury",          # UI側のトンマナ選択をここに渡す
    ) -> GenerationResult:
        """
        プロバイダーに関わらず同じ引数で呼べる統一メソッド。
        内部で各APIへの変換を行う。
        """
        full_prompt = self._build_prompt(prompt, style)

        if self.provider == "firefly":
            return self._call_firefly(full_prompt, width, height)
        elif self.provider == "gemini":
            return self._call_gemini(full_prompt, width, height)
        elif self.provider == "chatgpt":
            return self._call_chatgpt(full_prompt, width, height)
        else:
            raise ValueError(f"未対応のプロバイダー: {self.provider}")

    # ------------------------------------------------------------------ #
    # プロンプト生成（Claude APIでトンマナを自動変換する場所）
    # ------------------------------------------------------------------ #
    def _build_prompt(self, user_text: str, style: str) -> str:
        """
        ユーザーが入力したテキスト＋トンマナ指定を
        画像生成用の英語プロンプトに変換する。
        本番では Claude API を呼んで自動生成させる。
        """
        style_map = {
            "luxury":  "luxurious, deep crimson and black, gold accents, elegant nightclub",
            "pop":     "bright, vivid colors, playful, modern pop style",
            "cool":    "cool tones, minimal, stylish, monochrome with accent color",
            "girly":   "pastel pink, cute, feminine, soft lighting",
        }
        style_desc = style_map.get(style, style_map["luxury"])
        return (
            f"Professional advertising banner for a night entertainment venue. "
            f"{style_desc}. "
            f"Text overlay area on the right side. "
            f"No explicit content. High quality, photorealistic. "
            f"Context: {user_text}"
        )

    # ------------------------------------------------------------------ #
    # Adobe Firefly
    # ------------------------------------------------------------------ #
    def _call_firefly(self, prompt: str, width: int, height: int) -> GenerationResult:
        # Step1: アクセストークン取得
        token_res = requests.post(
            "https://ims-na1.adobelogin.com/ims/token/v3",
            data={
                "grant_type": "client_credentials",
                "client_id": self.FIREFLY_CLIENT_ID,
                "client_secret": self.FIREFLY_CLIENT_SECRET,
                "scope": "openid,AdobeID,firefly_enterprise",
            },
        )
        token_res.raise_for_status()
        access_token = token_res.json()["access_token"]

        # Step2: 画像生成
        res = requests.post(
            "https://firefly-api.adobe.io/v3/images/generate",
            headers={
                "Authorization": f"Bearer {access_token}",
                "x-api-key": self.FIREFLY_CLIENT_ID,
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt,
                "size": {"width": width, "height": height},
                "numVariations": 1,
                "contentClass": "photo",
            },
        )
        res.raise_for_status()
        data = res.json()
        image_url = data["outputs"][0]["image"]["url"]

        image_bytes = requests.get(image_url).content
        return GenerationResult(
            image_bytes=image_bytes,
            provider="firefly",
            prompt_used=prompt,
            width=width,
            height=height,
        )

    # ------------------------------------------------------------------ #
    # Google Gemini (Imagen 3)
    # ------------------------------------------------------------------ #
    def _call_gemini(self, prompt: str, width: int, height: int) -> GenerationResult:
        # Imagen 3 はアスペクト比指定のみ。1920:1536 = 5:4
        ratio = width / height
        if ratio >= 1.7:
            aspect = "16:9"
        elif ratio >= 1.2:
            aspect = "5:4"
        elif ratio <= 0.6:
            aspect = "9:16"
        else:
            aspect = "4:5"

        res = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict",
            headers={"Content-Type": "application/json"},
            params={"key": self.GEMINI_API_KEY},
            json={
                "instances": [{"prompt": prompt}],
                "parameters": {
                    "sampleCount": 1,
                    "aspectRatio": aspect,
                },
            },
        )
        res.raise_for_status()
        b64 = res.json()["predictions"][0]["bytesBase64Encoded"]
        image_bytes = base64.b64decode(b64)

        # 指定サイズに正確にリサイズ
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((width, height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        image_bytes = buf.getvalue()

        return GenerationResult(
            image_bytes=image_bytes,
            provider="gemini",
            prompt_used=prompt,
            width=width,
            height=height,
        )

    # ------------------------------------------------------------------ #
    # OpenAI DALL-E 3
    # ------------------------------------------------------------------ #
    def _call_chatgpt(self, prompt: str, width: int, height: int) -> GenerationResult:
        # DALL-E 3 は 1024x1024 / 1792x1024 / 1024x1792 のみ対応
        dalle_size = self._nearest_dalle_size(width, height)

        res = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {self.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": dalle_size,
                "response_format": "b64_json",
                "quality": "hd",
            },
        )
        res.raise_for_status()
        b64 = res.json()["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64)

        # 1920×1536に正確にリサイズ（DALL-E 3は固定サイズのため）
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((width, height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        image_bytes = buf.getvalue()

        return GenerationResult(
            image_bytes=image_bytes,
            provider="chatgpt",
            prompt_used=prompt,
            width=width,
            height=height,
        )

    @staticmethod
    def _nearest_dalle_size(width: int, height: int) -> str:
        """DALL-E 3 の対応サイズに丸める"""
        ratio = width / height
        if ratio > 1.3:
            return "1792x1024"
        elif ratio < 0.8:
            return "1024x1792"
        return "1024x1024"


# ------------------------------------------------------------------ #
# バナー合成（Pillow）— プロバイダー共通
# ------------------------------------------------------------------ #
def compose_banner(
    result: GenerationResult,
    text_lines: list[str],
    output_path: str,
) -> str:
    """
    生成した背景画像にテキストを合成してPNGで保存する。
    Pillowが必要: pip install Pillow
    """
    from PIL import Image, ImageDraw, ImageFont
    import io

    bg = Image.open(io.BytesIO(result.image_bytes)).convert("RGBA")
    bg = bg.resize((result.width, result.height), Image.LANCZOS)

    draw = ImageDraw.Draw(bg)

    # フォントはシステムフォントかfontsフォルダのものを指定
    try:
        font_large = ImageFont.truetype("fonts/NotoSansJP-Bold.ttf", 48)
        font_small = ImageFont.truetype("fonts/NotoSansJP-Regular.ttf", 28)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = font_large

    # テキストを右側に配置（背景画像の右40%エリア）
    x_start = int(result.width * 0.55)
    y = int(result.height * 0.3)

    for i, line in enumerate(text_lines):
        font = font_large if i == 0 else font_small
        draw.text((x_start, y), line, font=font, fill=(255, 255, 255, 230))
        y += 60 if i == 0 else 40

    bg = bg.convert("RGB")
    bg.save(output_path, "PNG", quality=95)
    print(f"保存完了: {output_path}")
    return output_path


# ------------------------------------------------------------------ #
# 動作確認用サンプル
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "chatgpt"
    print(f"プロバイダー: {provider} で生成テスト")

    router = AIProviderRouter(provider=provider)

    result = router.generate_image(
        prompt="高収入・完全日払い CLUB VENUS",
        width=1920,
        height=1536,
        style="luxury",
    )
    print(f"生成完了: {len(result.image_bytes):,} bytes")

    compose_banner(
        result=result,
        text_lines=["CLUB VENUS", "高収入・完全日払い", "TEL: 00-0000-0000"],
        output_path=f"output_banner_{provider}.png",
    )

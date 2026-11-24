import numpy as np

from .base import ImageFilter


class Grayscale(ImageFilter):
    """Преобразует изображение в оттенки серого"""

    def apply(self, image, **kwargs):
        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)

        gray = 0.114 * blue + 0.587 * green + 0.299 * red

        return np.clip(gray, 0, 255).astype(np.uint8)


class Antique(ImageFilter):
    """Придаёт изображению эффект старой фотографии с сепией и виньеткой."""

    def apply(self, image, vignette_strength=0.4, **kwargs):
        if not 0 <= vignette_strength <= 1:
            raise ValueError("vignette_strength должен быть от 0 до 1")

        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)

        result = np.empty_like(image, dtype=np.float32)

        result[:, :, 2] = (
            0.393 * red + 0.769 * green + 0.189 * blue
        )
        result[:, :, 1] = (
            0.349 * red + 0.686 * green + 0.168 * blue
        )
        result[:, :, 0] = (
            0.272 * red + 0.534 * green + 0.131 * blue
        )

        height, width = image.shape[:2]

        y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        distance = np.sqrt(x ** 2 + y ** 2)

        mask = np.clip((distance - 0.4) / 0.6, 0, 1)
        mask = mask ** 1.5 * vignette_strength

        result *= (1 - mask[:, :, None])

        return np.clip(result, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    """Имитирует выцветание изображения заданным цветом"""

    def apply(self, image, alpha=0.35, color=(235, 215, 175), **kwargs):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha должен быть от 0 до 1")

        fade_color = np.array(color[::-1], dtype=np.float32)
        original = image.astype(np.float32)

        result = (1 - alpha) * original + alpha * fade_color

        return np.clip(result, 0, 255).astype(np.uint8)




class Film(ImageFilter):

    def apply(self, image, strength=0.8, noise=3, **kwargs):
        if not 0 <= strength <= 1:
            raise ValueError("strength должен быть от 0 до 1")
        if noise < 0:
            raise ValueError("noise не может быть отрицательным")

        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)

        brightness = (
            0.114 * blue
            + 0.587 * green
            + 0.299 * red
        )

        contrast = np.clip(
            (brightness - 128) * 1.2 + 128,
            0,
            255
        )

        levels = np.array(
            [0, 50, 115, 185, 255],
            dtype=np.float32
        )

        blue_palette = np.array(
            [5, 15, 45, 90, 190],
            dtype=np.float32
        )

        green_palette = np.array(
            [3, 8, 30, 105, 220],
            dtype=np.float32
        )

        red_palette = np.array(
            [5, 25, 115, 220, 255],
            dtype=np.float32
        )

        infrared = np.empty_like(image, dtype=np.float32)

        infrared[:, :, 0] = np.interp(
            contrast, levels, blue_palette
        )
        infrared[:, :, 1] = np.interp(
            contrast, levels, green_palette
        )
        infrared[:, :, 2] = np.interp(
            contrast, levels, red_palette
        )

        maximum_other = np.maximum(red, green)
        blue_excess = blue - maximum_other

        saturation = (
            np.maximum(np.maximum(red, green), blue)
            - np.minimum(np.minimum(red, green), blue)
        ) / (np.maximum(np.maximum(red, green), blue) + 1.0)

        sky_mask = np.clip(
            (blue_excess - 5) / 60,
            0,
            1
        )

        sky_mask *= np.clip(
            (saturation - 0.1) / 0.3,
            0,
            1
        )

        infrared *= (1 - 0.8 * sky_mask[:, :, None])

        original = image.astype(np.float32)

        result = (
            original * (1 - strength)
            + infrared * strength
        )

        if noise > 0:
            rng = np.random.default_rng()
            grain = rng.normal(
                0,
                noise,
                image.shape[:2] + (1,)
            ).astype(np.float32)

            result += grain

        return np.clip(result, 0, 255).astype(np.uint8)

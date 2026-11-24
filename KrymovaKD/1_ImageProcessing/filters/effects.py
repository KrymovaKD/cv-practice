
import numpy as np

from .base import ImageFilter


class Matte(ImageFilter):
    """Создаёт белую мягкую овальную рамку по краям"""

    def apply(self, image, radius=0.58, strength=1.0, softness=2.2, **kwargs):
        if not 0 < radius < 1:
            raise ValueError("radius должен быть между 0 и 1")
        if not 0 <= strength <= 1:
            raise ValueError("strength должен быть от 0 до 1")
        if softness <= 0:
            raise ValueError("softness должен быть положительным")

        height, width = image.shape[:2]

        y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        distance = np.sqrt(x**2 + y**2)

        t = np.clip((distance - radius) / (1 - radius), 0, 1)
        mask = (t**softness * strength)[:, :, None]

        result = image.astype(np.float32) * (1 - mask) + 255 * mask

        return np.clip(result, 0, 255).astype(np.uint8)


def box_blur(array):
    """Усредняющее размытие двумерного массива 3x3."""
    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError("box_blur ожидает двумерный массив")

    padded = np.pad(array, 1, mode="edge")

    return (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

class Noise(ImageFilter):
    """Состаривание фотографии: зерно, пыль, царапины и яркая засветка."""

    def apply(self, image, amount=0.3, scratch_density=0.1, **kwargs):
        if amount < 0:
            raise ValueError("amount не может быть отрицательным")
        if not 0 <= scratch_density <= 1:
            raise ValueError("scratch_density должен быть от 0 до 1")

        rng = np.random.default_rng()
        result = image.astype(np.float32).copy()
        height, width = image.shape[:2]

        result *= 0.88

        grain = rng.normal(0, amount * 24, (height, width, 1))
        result += grain.astype(np.float32)

        light_scratches = np.zeros((height, width), dtype=np.float32)
        dark_scratches = np.zeros((height, width), dtype=np.float32)

        area = height * width / 1_000_000
        count = int(area * (500 + 1600 * scratch_density))

        directions = [
            rng.uniform(-0.18, 0.18),
            rng.uniform(1.35, 1.80),
            rng.uniform(-0.80, -0.45),
            rng.uniform(0.45, 0.80),
        ]

        for _ in range(count):
            x0 = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height))

            kind = rng.choice(
                ["short", "medium", "long"],
                p=[0.70, 0.22, 0.08]
            )

            if kind == "short":
                length = int(rng.integers(3, 12))
            elif kind == "medium":
                length = int(rng.integers(12, 35))
            else:
                length = int(rng.integers(
                    30, max(31, min(110, max(height, width) // 2))
                ))

            angle = float(rng.choice(directions) + rng.normal(0, 0.14))
            curve = rng.uniform(-2.0, 2.0)
            waves = rng.uniform(0.5, 1.8)
            phase = rng.uniform(0, 2 * np.pi)
            drift = rng.normal(0, 0.10)

            steps = max(length * 3, 10)
            mask = (
                light_scratches
                if rng.random() < 0.78
                else dark_scratches
            )
            brightness = rng.uniform(0.20, 0.65)
            previous_x, previous_y = None, None

            for i in range(steps):
                t = i / max(steps - 1, 1)

                bend = (
                    curve * np.sin(np.pi * t)
                    + 0.55 * np.sin(t * waves * 2 * np.pi + phase)
                    + drift * length * t
                )

                px = int(round(
                    x0 + length * t * np.cos(angle)
                    - bend * np.sin(angle)
                ))
                py = int(round(
                    y0 + length * t * np.sin(angle)
                    + bend * np.cos(angle)
                ))

                if not (0 <= px < width and 0 <= py < height):
                    previous_x, previous_y = None, None
                    continue

                if rng.random() < 0.055:
                    previous_x, previous_y = None, None
                    continue

                if previous_x is not None:
                    distance = max(
                        abs(px - previous_x),
                        abs(py - previous_y),
                        1
                    )

                    xs = np.rint(np.linspace(
                        previous_x, px, distance + 1
                    )).astype(int)
                    ys = np.rint(np.linspace(
                        previous_y, py, distance + 1
                    )).astype(int)

                    for j, (lx, ly) in enumerate(zip(xs, ys)):
                        if 0 <= lx < width and 0 <= ly < height:
                            local_brightness = brightness * (
                                0.78 + 0.22 * np.sin(
                                    np.pi * j / max(len(xs) - 1, 1)
                                )
                            )
                            mask[ly, lx] = max(
                                mask[ly, lx], local_brightness
                            )

                mask[py, px] = max(mask[py, px], brightness)
                previous_x, previous_y = px, py

        light_soft = box_blur(light_scratches)
        dark_soft = box_blur(dark_scratches)

        light = light_scratches * 0.72 + light_soft * 0.28
        dark = dark_scratches * 0.70 + dark_soft * 0.30

        texture = rng.uniform(
            0.82, 1.08, size=(height, width)
        ).astype(np.float32)
        light *= texture

        result[:, :, 0] += light * 58
        result[:, :, 1] += light * 64
        result[:, :, 2] += light * 68

        result -= dark[:, :, None] * 28

        dust_count = int(height * width * 0.00022)

        for _ in range(dust_count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            radius = int(rng.choice([1, 1, 1, 1, 2, 3]))
            brightness = float(rng.uniform(15, 75))

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px, py = x + dx, y + dy

                    if not (0 <= px < width and 0 <= py < height):
                        continue

                    distance = np.sqrt(dx * dx + dy * dy)

                    if distance <= radius:
                        value = brightness * (
                            1 - distance / (radius + 1)
                        )
                        result[py, px, 0] += value * 0.82
                        result[py, px, 1] += value * 0.92
                        result[py, px, 2] += value

        yy, xx = np.mgrid[0:height, 0:width]
        x_norm = xx / max(width - 1, 1)
        y_norm = yy / max(height - 1, 1)

        distance = np.sqrt(
            (x_norm / 0.42) ** 2
            + ((1 - y_norm) / 0.38) ** 2
        )
        leak = np.clip(1 - distance, 0, 1) ** 2

        extra_leak = 0.65 * np.exp(-(
            (x_norm / 0.28) ** 2
            + ((1 - y_norm) / 0.22) ** 2
        ))

        leak = np.maximum(leak, extra_leak).astype(np.float32)

        leak = box_blur(leak)
        leak = box_blur(leak)
        leak = box_blur(leak)

        result[:, :, 0] += leak * 45
        result[:, :, 1] += leak * 110
        result[:, :, 2] += leak * 190

        return np.clip(result, 0, 255).astype(np.uint8)





class Neon(ImageFilter):

    @staticmethod
    def fast_blur(array):
        """Быстрое сглаживание 3x3 для подготовки изображения."""
        padded = np.pad(array, 1, mode="edge")

        return (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

    @staticmethod
    def gaussian_blur(array, sigma):
        """Быстрая аппроксимация гауссова размытия тремя box blur."""

        if sigma <= 0:
            return array.astype(np.float32).copy()

        result = array.astype(np.float32)
        radius = max(1, int(round(sigma)))
        window = 2 * radius + 1

        for _ in range(3):

            padded = np.pad(
                result,
                ((0, 0), (radius, radius), (0, 0)),
                mode="edge"
            )

            cumulative = np.cumsum(
                padded,
                axis=1,
                dtype=np.float32
            )

            cumulative = np.pad(
                cumulative,
                ((0, 0), (1, 0), (0, 0)),
                mode="constant"
            )

            horizontal = (
                cumulative[:, window:, :] -
                cumulative[:, :-window, :]
            ) / window

            # Вертикальное размытие
            padded = np.pad(
                horizontal,
                ((radius, radius), (0, 0), (0, 0)),
                mode="edge"
            )

            cumulative = np.cumsum(
                padded,
                axis=0,
                dtype=np.float32
            )

            cumulative = np.pad(
                cumulative,
                ((1, 0), (0, 0), (0, 0)),
                mode="constant"
            )

            result = (
                cumulative[window:, :, :] -
                cumulative[:-window, :, :]
            ) / window

        return result

    def apply(self, image, strength=2.0, blur=3.5, **kwargs):
        """Создает неоновое свечение по контурам изображения."""

        image = image.astype(np.float32)

        gray = (
            0.114 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.299 * image[:, :, 2]
        )

        gray = self.fast_blur(gray)
        gray = self.fast_blur(gray)

        padded = np.pad(gray, 1, mode="edge")

        gx = (
            -padded[:-2, :-2] + padded[:-2, 2:] -
            2 * padded[1:-1, :-2] +
            2 * padded[1:-1, 2:] -
            padded[2:, :-2] + padded[2:, 2:]
        )

        gy = (
            -padded[:-2, :-2] -
            2 * padded[:-2, 1:-1] -
            padded[:-2, 2:] +
            padded[2:, :-2] +
            2 * padded[2:, 1:-1] +
            padded[2:, 2:]
        )

        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        low = np.percentile(magnitude, 90)
        high = np.percentile(magnitude, 99.5)

        if high > low:
            edges = np.clip(
                (magnitude - low) / (high - low),
                0,
                1
            )
        else:
            edges = np.zeros_like(magnitude)

        edges = edges ** 1.5

        glow = np.zeros_like(image, dtype=np.float32)

        glow[:, :, 0] = 220 * edges
        glow[:, :, 1] = 15 * edges
        glow[:, :, 2] = 255 * edges

        near_glow = self.gaussian_blur(
            glow,
            sigma=3.5
        )

        middle_glow = self.gaussian_blur(
            glow,
            sigma=max(2.0, blur * 2.5)
        )

        wide_glow = self.gaussian_blur(
            glow,
            sigma=max(4.0, blur * 5.0)
        )

        core = np.zeros_like(image, dtype=np.float32)

        core[:, :, 0] = 255 * edges
        core[:, :, 1] = 175 * edges
        core[:, :, 2] = 255 * edges

        base = image * 0.65

        result = (
            base
            + wide_glow * strength * 5.0
            + middle_glow * strength * 2.5
            + near_glow * strength * 1.5
            + core * strength
        )

        return np.clip(result, 0, 255).astype(np.uint8)







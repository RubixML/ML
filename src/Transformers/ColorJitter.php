<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

use function Rubix\ML\iterator_first;
use function abs;
use function fmod;
use function rand;
use function imagealphablending;
use function imagecolorallocatealpha;
use function imagecolorat;
use function imagecreatetruecolor;
use function imageistruecolor;
use function imagesavealpha;
use function imagesetpixel;
use function imagesx;
use function imagesy;
use function imagepalettetotruecolor;
use function max;
use function min;
use function round;

use const PHP_INT_MAX;

/**
 * Color Jitter
 *
 * Adds random jitter to the brightness, contrast, saturation, and hue of images.
 * Non-image feature columns are ignored.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ColorJitter implements Transformer, Stringable
{
    /**
     * The maximum brightness adjustment factor.
     *
     * @var float
     */
    protected float $brightness;

    /**
     * The maximum contrast adjustment factor.
     *
     * @var float
     */
    protected float $contrast;

    /**
     * The maximum saturation adjustment factor.
     *
     * @var float
     */
    protected float $saturation;

    /**
     * The maximum hue shift in degrees.
     *
     * @var float
     */
    protected float $hue;

    /**
     * Convert an RGB color to HSV.
     *
     * @param int $r
     * @param int $g
     * @param int $b
     * @return array{float,float,float}
     */
    protected static function rgbToHsv(int $r, int $g, int $b) : array
    {
        $rNorm = $r / 255.0;
        $gNorm = $g / 255.0;
        $bNorm = $b / 255.0;

        $maxVal = max($rNorm, $gNorm, $bNorm);
        $minVal = min($rNorm, $gNorm, $bNorm);

        $delta = $maxVal - $minVal;

        $h = 0.0;
        $s = 0.0;
        $v = $maxVal;

        if ($delta > 0.0) {
            $s = $maxVal === 0.0 ? 0.0 : $delta / $maxVal;

            if ($maxVal === $rNorm) {
                $h = 60.0 * fmod(($gNorm - $bNorm) / $delta, 6.0);
            } elseif ($maxVal === $gNorm) {
                $h = 60.0 * (($bNorm - $rNorm) / $delta + 2.0);
            } elseif ($maxVal === $bNorm) {
                $h = 60.0 * (($rNorm - $gNorm) / $delta + 4.0);
            }
        }

        if ($h < 0.0) {
            $h += 360.0;
        }

        return [$h, $s, $v];
    }

    /**
     * Convert an HSV color to RGB.
     *
     * @param float $h
     * @param float $s
     * @param float $v
     * @return array{int,int,int}
     */
    protected static function hsvToRgb(float $h, float $s, float $v) : array
    {
        $hNorm = $h;

        while ($hNorm >= 360.0) {
            $hNorm -= 360.0;
        }

        while ($hNorm < 0.0) {
            $hNorm += 360.0;
        }

        $c = $v * $s;
        $x = $c * (1.0 - abs(fmod($hNorm / 60.0, 2.0) - 1.0));
        $m = $v - $c;

        $r1 = 0.0;
        $g1 = 0.0;
        $b1 = 0.0;

        if ($hNorm < 60.0) {
            $r1 = $c;
            $g1 = $x;
            $b1 = 0.0;
        } elseif ($hNorm < 120.0) {
            $r1 = $x;
            $g1 = $c;
            $b1 = 0.0;
        } elseif ($hNorm < 180.0) {
            $r1 = 0.0;
            $g1 = $c;
            $b1 = $x;
        } elseif ($hNorm < 240.0) {
            $r1 = 0.0;
            $g1 = $x;
            $b1 = $c;
        } elseif ($hNorm < 300.0) {
            $r1 = $x;
            $g1 = 0.0;
            $b1 = $c;
        } else {
            $r1 = $c;
            $g1 = 0.0;
            $b1 = $x;
        }

        $r = (int) round(($r1 + $m) * 255.0);
        $g = (int) round(($g1 + $m) * 255.0);
        $b = (int) round(($b1 + $m) * 255.0);

        $r = max(0, min(255, $r));
        $g = max(0, min(255, $g));
        $b = max(0, min(255, $b));

        return [$r, $g, $b];
    }

    /**
     * @param float $brightness
     * @param float $contrast
     * @param float $saturation
     * @param float $hue
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $brightness = 0.2,
        float $contrast = 0.2,
        float $saturation = 0.2,
        float $hue = 0.0
    ) {
        if ($brightness < 0.0) {
            throw new InvalidArgumentException('Brightness must be greater'
                . ' than or equal to 0.');
        }

        if ($contrast < 0.0) {
            throw new InvalidArgumentException('Contrast must be greater'
                . ' than or equal to 0.');
        }

        if ($saturation < 0.0) {
            throw new InvalidArgumentException('Saturation must be greater'
                . ' than or equal to 0.');
        }

        if ($hue < 0.0) {
            throw new InvalidArgumentException('Hue must be greater'
                . ' than or equal to 0.');
        }

        $this->brightness = $brightness;
        $this->contrast = $contrast;
        $this->saturation = $saturation;
        $this->hue = $hue;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        if (empty($samples)) {
            return;
        }

        $types = array_map([DataType::class, 'detect'], iterator_first($samples));

        $types = array_filter($types, fn ($type) => $type->isImage());

        $columns = array_keys($types);

        if (empty($columns)) {
            return;
        }

        array_walk($samples, [$this, 'jitter'], $columns);
    }

    /**
     * Jitter the images within a sample.
     *
     * @param array<mixed> $sample
     * @param int $index
     * @param array<int> $columns
     */
    protected function jitter(array &$sample, int $index, array $columns) : void
    {
        foreach ($columns as $column) {
            $image = $sample[$column];

            if (!imageistruecolor($image)) {
                imagepalettetotruecolor($image);
            }

            $width = imagesx($image);
            $height = imagesy($image);

            $out = imagecreatetruecolor($width, $height);

            imagealphablending($out, false);
            imagesavealpha($out, true);

            $r1 = rand(-PHP_INT_MAX, PHP_INT_MAX) / PHP_INT_MAX;
            $r2 = rand(-PHP_INT_MAX, PHP_INT_MAX) / PHP_INT_MAX;
            $r3 = rand(-PHP_INT_MAX, PHP_INT_MAX) / PHP_INT_MAX;
            $r4 = rand(-PHP_INT_MAX, PHP_INT_MAX) / PHP_INT_MAX;

            $brightnessFactor = 1.0 + $this->brightness * $r1;
            $contrastFactor = 1.0 + $this->contrast * $r2;
            $saturationFactor = 1.0 + $this->saturation * $r3;
            $hueShift = $this->hue * $r4;

            if ($brightnessFactor < 0.0) {
                $brightnessFactor = 0.0;
            }

            for ($y = 0; $y < $height; ++$y) {
                for ($x = 0; $x < $width; ++$x) {
                    $pixel = imagecolorat($image, $x, $y);

                    if ($pixel === false) {
                        continue;
                    }

                    $a = ($pixel >> 24) & 0x7F;
                    $r = ($pixel >> 16) & 0xFF;
                    $g = ($pixel >> 8) & 0xFF;
                    $b = $pixel & 0xFF;

                    if ($this->brightness > 0.0) {
                        $r = (int) round($r * $brightnessFactor);
                        $g = (int) round($g * $brightnessFactor);
                        $b = (int) round($b * $brightnessFactor);

                        $r = max(0, min(255, $r));
                        $g = max(0, min(255, $g));
                        $b = max(0, min(255, $b));
                    }

                    if ($this->contrast > 0.0) {
                        $mean = ($r + $g + $b) / 3.0;

                        $r = (int) round($mean + ($r - $mean) * $contrastFactor);
                        $g = (int) round($mean + ($g - $mean) * $contrastFactor);
                        $b = (int) round($mean + ($b - $mean) * $contrastFactor);

                        $r = max(0, min(255, $r));
                        $g = max(0, min(255, $g));
                        $b = max(0, min(255, $b));
                    }

                    if ($this->saturation > 0.0) {
                        [$hsvH, $hsvS, $hsvV] = self::rgbToHsv($r, $g, $b);

                        $hsvSNew = $hsvS * $saturationFactor;

                        if ($hsvSNew > 1.0) {
                            $hsvSNew = 1.0;
                        }

                        if ($hsvSNew < 0.0) {
                            $hsvSNew = 0.0;
                        }

                        [$r, $g, $b] = self::hsvToRgb($hsvH, $hsvSNew, $hsvV);
                    }

                    if ($this->hue > 0.0) {
                        [$hsvH, $hsvS, $hsvV] = self::rgbToHsv($r, $g, $b);

                        $hsvHNew = $hsvH + $hueShift;

                        while ($hsvHNew >= 360.0) {
                            $hsvHNew -= 360.0;
                        }

                        while ($hsvHNew < 0.0) {
                            $hsvHNew += 360.0;
                        }

                        [$r, $g, $b] = self::hsvToRgb($hsvHNew, $hsvS, $hsvV);
                    }

                    $color = imagecolorallocatealpha($out, $r, $g, $b, $a);

                    if ($color !== false) {
                        imagesetpixel($out, $x, $y, $color);
                    }
                }
            }

            $sample[$column] = $out;
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "ColorJitter (brightness: {$this->brightness}, contrast: {$this->contrast},"
            . " saturation: {$this->saturation}, hue: {$this->hue})";
    }
}

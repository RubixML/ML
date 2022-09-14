<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function rand;
use function array_walk;
use function getrandmax;

/**
 * Randomized Image Rotator
 *
 * Randomly rotates the image between 0 and a given number of max degrees.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Stylianos Tzourelis
 */
class RandomizedImageRotator implements Transformer
{
    /**
     * The angle in degrees to rotate an image anti-clockwise.
     *
     * @var float
     */
    protected float $maxDegrees;

    /**
     * @param float $maxDegrees
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $maxDegrees = 360.0)
    {
        ExtensionIsLoaded::with('gd')->check();

        if ($maxDegrees < 0.0 or $maxDegrees > 360.0) {
            throw new InvalidArgumentException('Degrees must be '
                . " greater than 0, and less than 360 and $maxDegrees given.");
        }

        $this->maxDegrees = $maxDegrees;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
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
        array_walk($samples, [$this, 'rotateAndCrop']);
    }

    /**
     * Randomly rotates and crops the images in a sample to their original size.
     *
     * @internal
     *
     * @param list<mixed> $sample
     */
    public function rotateAndCrop(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (DataType::detect($value)->isImage()) {
                $degrees = $this->randomRotationAngle();

                $originalWidth = imagesx($value);
                $originalHeight = imagesy($value);

                $rotated = imagerotate($value, $degrees, 0);

                if ($rotated) {
                    $newHeight = imagesy($rotated);
                    $newWidth = imagesx($rotated);

                    if ($originalHeight !== $newHeight or $originalWidth !== $newWidth) {
                        $rotated = imagecrop($rotated, [
                            'x' => $newWidth / 2 - $originalWidth / 2,
                            'y' => $newHeight / 2 - $originalHeight / 2,
                            'width' => $originalWidth,
                            'height' => $originalHeight
                        ]);
                    }

                    $value = $rotated;
                }
            }
        }
    }

    /**
     * Return a random rotation angle in degrees.
     *
     * @internal
     *
     * @return float
     */
    public function randomRotationAngle() : float
    {
        $phi = getrandmax() / $this->maxDegrees;

        $mHat = (int) ($this->maxDegrees * $phi);

        return rand(0, $mHat) / $phi;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Randomized Image Rotator (maxDegrees: {$this->maxDegrees})";
    }
}

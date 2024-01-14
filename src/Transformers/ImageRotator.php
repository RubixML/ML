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
class ImageRotator implements Transformer
{
    /**
     * The color of the area of the image filled in after rotation and cropping.
     *
     * @var int
     */
    protected const FILL_COLOR = 0;

    /**
     * The number of degrees to rotate the image.
     *
     * @var float
     */
    protected float $offset;

    /**
     * The amount of random jitter to add to the rotation.
     *
     * @var float
     */
    protected float $jitter;

    /**
     * @param float $offset
     * @param float $jitter
     * @throws InvalidArgumentException
     */
    public function __construct(float $offset, float $jitter = 0.0)
    {
        ExtensionIsLoaded::with('gd')->check();

        if ($offset < -360.0 or $offset > 360.0) {
            throw new InvalidArgumentException('Offset must be '
                . " greater than -360, and less than 360 and $offset given.");
        }

        if ($jitter < 0.0 or $jitter > 1.0) {
            throw new InvalidArgumentException('Jitter must be '
                . " greater than 0, and less than 1 and $jitter given.");
        }

        $this->offset = $offset;
        $this->jitter = $jitter;
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
    protected function rotateAndCrop(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (DataType::detect($value)->isImage()) {
                $degrees = $this->rotationAngle();

                $originalWidth = imagesx($value);
                $originalHeight = imagesy($value);

                $rotated = imagerotate($value, $degrees, self::FILL_COLOR);

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
     * Return an angle with a given offset with random jitter in degrees between 0 and 360.
     *
     * @return float
     */
    protected function rotationAngle() : float
    {
        $maxDegrees = $this->jitter * 180.0;

        $phi = getrandmax() / $maxDegrees;

        $mHat = intval($maxDegrees * $phi);

        $jitter = rand(-$mHat, $mHat) / $phi;

        $angle = $this->offset + $jitter;

        while ($angle < 0.0) {
            $angle += 360.0;
        }

        while ($angle >= 360.0) {
            $angle -= 360.0;
        }

        return $angle;
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
        return "Image Rotator (offset: {$this->offset}, jitter: {$this->jitter})";
    }
}

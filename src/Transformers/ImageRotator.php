<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\iterator_first;
use function rand;
use function array_keys;
use function array_walk;
use function array_filter;
use function array_map;
use function getrandmax;

/**
 * Randomized Image Rotator
 *
 * Rotates an image by a given offset angle and random jitter. The rotated image is then resized
 * back to its original dimensions. Permutations such as these are useful for training computer
 * vision models that are robust to rotation and small variations in orientation.
 *
 * > **Note:** The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Stylianos Tzourelis
 */
class ImageRotator implements Transformer
{
    /**
     * The color of the area of the image filled in after rotation.
     *
     * @var int
     */
    protected const FILL_COLOR = 0;

    /**
     * The offset angle in degrees to rotate before applying random jitter.
     *
     * @var float
     */
    protected float $offset;

    /**
     * The amount of random jitter to add to the rotation in either direction.
     *
     * @var float
     */
    protected float $jitter;

    /**
     * @param float $offset
     * @param float $jitter
     * @throws InvalidArgumentException
     */
    public function __construct(float $offset = 0.0, float $jitter = 0.2)
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

        array_walk($samples, [$this, 'rotateAndCrop'], $columns);
    }

    /**
     * Randomly rotates the images in a sample and resizes them back to their original size.
     *
     * @internal
     *
     * @param array<mixed> $sample
     * @param int $index
     * @param list<int> $columns
     * @throws RuntimeException
     */
    protected function rotateAndCrop(array &$sample, int $index, array $columns) : void
    {
        foreach ($columns as $column) {
            $image = $sample[$column];

            $degrees = $this->rotationAngle();

            $originalWidth = imagesx($image);
            $originalHeight = imagesy($image);

            $rotated = imagerotate($image, $degrees, self::FILL_COLOR);

            if ($rotated) {
                $newHeight = imagesy($rotated);
                $newWidth = imagesx($rotated);

                if ($originalHeight !== $newHeight or $originalWidth !== $newWidth) {
                    $resized = imagecreatetruecolor($originalWidth, $originalHeight);

                    if (!$resized) {
                        throw new RuntimeException('Could not create placeholder image.');
                    }

                    $success = imagecopyresampled(
                        $resized,
                        $rotated,
                        0,
                        0,
                        0,
                        0,
                        $originalWidth,
                        $originalHeight,
                        $newWidth,
                        $newHeight
                    );

                    if (!$success) {
                        throw new RuntimeException('Failed to resize image back to its original size.');
                    }

                    $rotated = $resized;
                }

                $sample[$column] = $rotated;
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

        if ($maxDegrees === 0.0) {
            $jitter = 0.0;
        } else {
            $phi = getrandmax() / $maxDegrees;

            $mHat = intval($maxDegrees * $phi);

            $jitter = rand(-$mHat, $mHat) / $phi;
        }

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

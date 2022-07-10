<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Image Random Rotation
 *
 * Rotates the image between 0 and a given number of degrees
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Stylianos Tzourelis
 */
class ImageRandomRotationer implements Transformer
{
    protected int $degrees;

    /**
     * @param int $degrees
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $degrees = 30)
    {
        ExtensionIsLoaded::with('gd')->check();

        if ($degrees < 1 or $degrees > 360) {
            throw new InvalidArgumentException('Degrees must be '
                . " greater than 1, and less than 360 and $degrees given.");
        }

        $this->degrees = $degrees;
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
        array_walk($samples, [$this, 'rotate']);
    }

    /**
     * rotates the images in a sample.
     *
     * @param list<mixed> $sample
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function rotate(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (DataType::detect($value)->isImage()) {
                $originalWidth = imagesx($value);
                $originalHeight = imagesy($value);

                $value = imagerotate($value, $this->getRotationDegrees(), 0);
                $newHeight = imagesy($value);
                $newWidth = imagesx($value);

                if ($originalHeight !== $newHeight || $originalWidth !== $newWidth) {
                    $value = imagecrop($value, [
                        'x' => $newWidth / 2 - $originalWidth / 2,
                        'y' => $newHeight / 2 - $originalHeight / 2,
                        'width' => $originalWidth,
                        'height' => $originalHeight
                    ]);
                }
            }
        }
    }

    public function getRotationDegrees() : int
    {
        return mt_rand(0, $this->degrees);
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
        return "Image Random Rotationer with a maximum of {$this->degrees} degrees.";
    }
}

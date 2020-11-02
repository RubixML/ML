<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use InvalidArgumentException;
use RuntimeException;
use Stringable;

/**
 * Image Resizer
 *
 * Image Resizer fits (scales and crops) images in a dataset to a user-specified width and height.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ImageResizer implements Transformer, Stringable
{
    /**
     * The width to fit the image to.
     *
     * @var int
     */
    protected $width;

    /**
     * The height to fit the image to.
     *
     * @var int
     */
    protected $height;

    /**
     * The ratio of width to height.
     *
     * @var float
     */
    protected $ratio;

    /**
     * @param int $width
     * @param int $height
     * @throws \InvalidArgumentException
     */
    public function __construct(int $width = 32, int $height = 32)
    {
        if (!extension_loaded('gd')) {
            throw new RuntimeException('GD extension is not loaded'
                . ', check PHP configuration.');
        }

        if ($width < 1 or $height < 1) {
            throw new InvalidArgumentException('Width and height must be'
                . " greater than 0, $width and $height given.");
        }

        $this->width = $width;
        $this->height = $height;
        $this->ratio = $width / $height;
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
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$value) {
                if (DataType::detect($value)->isImage()) {
                    $width = imagesx($value);
                    $height = imagesy($value);

                    if ($width === $this->width and $height === $this->height) {
                        continue 1;
                    }

                    if ($width / $height < $this->ratio) {
                        $srcW = $width;
                        $srcH = (int) ceil(($srcW * $this->height) / $this->width);
                        $srcY = (int) ceil(0.5 * ($height - $srcH));
                        $srcX = 0;
                    } else {
                        $srcH = $height;
                        $srcW = (int) ceil(($srcH * $this->width) / $this->height);
                        $srcX = (int) ceil(0.5 * ($width - $srcW));
                        $srcY = 0;
                    }

                    $resized = imagecreatetruecolor($this->width, $this->height);

                    if (!$resized) {
                        throw new RuntimeException('Could not create placeholder image.');
                    }

                    imagecopyresampled(
                        $resized,
                        $value,
                        0,
                        0,
                        $srcX,
                        $srcY,
                        $this->width,
                        $this->height,
                        $srcW,
                        $srcH
                    );

                    $value = $resized;
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Image Resizer (width: {$this->width}, height: {$this->height})";
    }
}

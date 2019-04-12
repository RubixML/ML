<?php

namespace Rubix\ML\Transformers;

use Intervention\Image\ImageManager;
use InvalidArgumentException;

/**
 * Image Resizer
 *
 * The Image Resizer scales and crops images to a user specified width and height.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ImageResizer implements Transformer
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
     * The Intervention image manager instance.
     *
     * @var \Intervention\Image\ImageManager
     */
    protected $intervention;

    /**
     * @param int $width
     * @param int $height
     * @param string $driver
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $width = 32,
        int $height = 32,
        string $driver = 'gd'
    ) {
        if ($width < 1 or $height < 1) {
            throw new InvalidArgumentException('Width and height must be'
                . " greater than 1 pixel, $width and $height given.");
        }

        $this->width = $width;
        $this->height = $height;
        $this->intervention = new ImageManager(['driver' => $driver]);
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($sample as $column => &$value) {
                if (is_resource($value) ? get_resource_type($value) === 'gd' : false) {
                    $image = $this->intervention->make($value);

                    $resize = $image->getWidth() !== $this->width
                        and $image->getHeight() !== $this->height;

                    if ($resize) {
                        $image = $image->fit($this->width, $this->height);
                    }

                    $value = $image->getCore();
                }
            }
        }
    }
}

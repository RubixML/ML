<?php

namespace Rubix\ML\Transformers;

use Intervention\Image\ImageManager;
use InvalidArgumentException;
use RuntimeException;

/**
 * Image Vectorizer
 *
 * Image Vectorizer takes images (as PHP Resources) and converts them into a
 * flat vector of raw color channel data. Scaling and cropping is handled
 * automatically by Intervention Image for PHP.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ImageVectorizer implements Transformer
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
     * The number of color channels to encode.
     *
     * @var int
     */
    protected $channels;

    /**
     * The Intervention image manager instance.
     *
     * @var \Intervention\Image\ImageManager
     */
    protected $intervention;

    /**
     * @param  array  $size
     * @param  bool  $rgb
     * @param  string  $driver
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return void
     */
    public function __construct(array $size = [32, 32], bool $rgb = true, string $driver = 'gd')
    {
        if (!extension_loaded('gd')) {
            throw new RuntimeException('GD extension is not loaded, check'
                . ' PHP configuration.');
        }

        $size = array_values($size);

        if (count($size) !== 2) {
            throw new InvalidArgumentException('Size must contain width and'
                . ' height but ' . count($size) . ' dimensions given.');
        }

        if (!is_int($size[0]) and !is_int($size[1])) {
            throw new InvalidArgumentException('Width and height must be'
                . ' integers, ' .  gettype($size[0])  . ' and '
                . gettype($size[1]) . ' given.');
        }

        if ($size[0] < 1 or $size[1] < 1) {
            throw new InvalidArgumentException('Width and height must be'
                . " greater than 1 pixel, $size[0] and $size[1] given.");
        }

        $this->width = $size[0];
        $this->height = $size[1];
        $this->channels = $rgb ? 3 : 1;
        $this->intervention = new ImageManager(['driver' => $driver]);
    }

    /**
     * Return the dimensionality of the vector that gets encoded.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return $this->width * $this->height * $this->channels;
    }

    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($sample as $column => $feature) {
                if (is_resource($feature)) {
                    $image = $this->intervention->make($feature);

                    $resize = $image->getWidth() !== $this->width
                        and $image->getHeight() !== $this->height;

                    if ($resize) {
                        $image = $image->fit($this->width, $this->height);
                    }

                    if ($this->channels === 1) {
                        $image = $image->greyscale();
                    }

                    $vectors[] = $this->vectorize($image->getCore());

                    $image->destroy();

                    unset($sample[$column]);
                }
            }

            $sample = array_merge($sample, ...$vectors);
        }
    }

    /**
     * Convert an image into a vector of raw color channel data.
     *
     * @param  mixed  $image
     * @throws \InvalidArgumentException
     * @return array
     */
    public function vectorize($image) : array
    {
        if (!is_resource($image)) {
            throw new InvalidArgumentException('Input must be a resource.');
        }

        $vector = [];

        for ($x = 0; $x < $this->width; $x++) {
            for ($y = 0; $y < $this->height; $y++) {
                $pixel = imagecolorsforindex($image, imagecolorat($image, $x, $y));

                $pixel = array_slice($pixel, 0, $this->channels);

                $vector = array_merge($vector, array_values($pixel));
            }
        }

        return $vector;
    }
}

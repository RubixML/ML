<?php

namespace Rubix\ML\Transformers;

use InvalidArgumentException;
use RuntimeException;

/**
 * Image Vectorizer
 *
 * Image Vectorizer takes images and converts them into a flat vector
 * of raw color channel data.
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
     * The number of color channels to encode.
     *
     * @var int
     */
    protected $channels;

    /**
     * The the number of bits to encode after the first 8.
     *
     * @var int
     */
    protected $mu;

    /**
     * @param int $channels
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(int $channels = 3)
    {
        if (!extension_loaded('gd')) {
            throw new RuntimeException('GD extension is not loaded, check'
                . ' PHP configuration.');
        }

        if ($channels < 1 or $channels > 3) {
            throw new InvalidArgumentException('The number of channels must'
                . " be between 1 and 3, $channels given.");
        }

        $this->channels = $channels;
        $this->mu = ($channels - 1) * 8;
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

            foreach ($sample as $column => $image) {
                if (is_resource($image) ? get_resource_type($image) === 'gd' : false) {
                    $width = imagesx($image);
                    $height = imagesy($image);

                    $vector = [];

                    for ($x = 0; $x < $width; $x++) {
                        for ($y = 0; $y < $height; $y++) {
                            $pixel = imagecolorat($image, $x, $y);

                            $vector[] = $pixel & 0xFF;
            
                            for ($i = 8; $i <= $this->mu; $i *= 2) {
                                $vector[] = ($pixel >> $i) & 0xFF;
                            }
                        }
                    }

                    unset($sample[$column]);

                    imagedestroy($image);

                    $vectors[] = $vector;
                }
            }

            $sample = array_merge($sample, ...$vectors);
        }
    }
}

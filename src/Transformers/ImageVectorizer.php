<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
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

        if ($channels < 1 or $channels > 4) {
            throw new InvalidArgumentException('The number of channels must'
                . " be between 1 and 4, $channels given.");
        }

        $this->channels = $channels;
        $this->mu = ($channels - 1) * 8;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($sample as $column => $value) {
                if (DataType::isImage($value)) {
                    $width = imagesx($value);
                    $height = imagesy($value);

                    $vector = [];

                    for ($x = 0; $x < $width; ++$x) {
                        for ($y = 0; $y < $height; ++$y) {
                            $pixel = imagecolorat($value, $x, $y);

                            $vector[] = $pixel & 0xFF;
            
                            for ($i = 8; $i <= $this->mu; $i *= 2) {
                                $vector[] = ($pixel >> $i) & 0xFF;
                            }
                        }
                    }

                    unset($sample[$column]);

                    imagedestroy($value);

                    $vectors[] = $vector;
                }
            }

            $sample = array_merge($sample, ...$vectors);
        }
    }
}

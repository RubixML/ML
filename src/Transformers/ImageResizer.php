<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\iterator_first;
use function array_map;
use function array_filter;
use function array_keys;
use function array_walk;
use function ceil;

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
class ImageResizer implements Transformer
{
    /**
     * The width to fit the image to.
     *
     * @var int
     */
    protected int $width;

    /**
     * The height to fit the image to.
     *
     * @var int
     */
    protected int $height;

    /**
     * The ratio of width to height.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * @param int $width
     * @param int $height
     * @throws InvalidArgumentException
     */
    public function __construct(int $width = 32, int $height = 32)
    {
        ExtensionIsLoaded::with('gd')->check();

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

        array_walk($samples, [$this, 'resize'], $columns);
    }

    /**
     * resize the images in a sample.
     *
     * @param array<mixed> $sample
     * @param int $index
     * @param list<int> $columns
     * @throws RuntimeException
     */
    public function resize(array &$sample, int $index, array $columns) : void
    {
        foreach ($columns as $column) {
            $image = $sample[$column];

            $width = imagesx($image);
            $height = imagesy($image);

            if ($width === $this->width and $height === $this->height) {
                continue;
            }

            if ($width / $height < $this->ratio) {
                $w = $width;
                $h = (int) ceil(($width * $this->height) / $this->width);

                $x = 0;
                $y = (int) ceil(0.5 * ($height - $h));
            } else {
                $w = (int) ceil(($height * $this->width) / $this->height);
                $h = $height;

                $x = (int) ceil(0.5 * ($width - $w));
                $y = 0;
            }

            $resized = imagecreatetruecolor($this->width, $this->height);

            if (!$resized) {
                throw new RuntimeException('Could not create placeholder image.');
            }

            $success = imagecopyresampled($resized, $image, 0, 0, $x, $y, $this->width, $this->height, $w, $h);

            if (!$success) {
                throw new RuntimeException('Failed to resize image.');
            }

            $sample[$column] = $resized;
        }
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
        return "Image Resizer (width: {$this->width}, height: {$this->height})";
    }
}

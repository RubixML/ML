<?php

namespace Rubix\ML\Clusterers\Seeders;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function array_slice;
use function array_unique;
use function array_values;

/**
 * Preset
 *
 * Generates centroids from a list of presets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Preset implements Seeder
{
    /**
     * A list of predefined cluster centroids to sample from.
     *
     * @var list<list<string|int|float>>
     */
    protected array $centroids;

    /**
     * The dimensionality of the predefined centroids.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * @param array<(string|int|float)[]> $centroids
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $centroids)
    {
        if (empty($centroids)) {
            throw new InvalidArgumentException('Number of centroids'
                . ' must be greater than 1, 0 given.');
        }

        $dimensions = count(current($centroids));

        $centroids = array_unique($centroids, SORT_REGULAR);

        foreach ($centroids as &$centroid) {
            if (count($centroid) !== $dimensions) {
                throw new InvalidArgumentException('Centroid must'
                    . " have $dimensions dimensions, "
                    . count($centroid) . ' given.');
            }

            $centroid = array_values($centroid);
        }

        $this->centroids = array_values($centroids);
        $this->dimensions = $dimensions;
    }

    /**
     * Seed k cluster centroids from a dataset.
     *
     * @internal
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<list<string|int|float>>
     */
    public function seed(Dataset $dataset, int $k) : array
    {
        DatasetHasDimensionality::with($dataset, $this->dimensions)->check();

        if ($k > count($this->centroids)) {
            throw new RuntimeException('Not enough unique'
                . " presets to generate $k centroids.");
        }

        return array_slice($this->centroids, 0, $k);
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
        return 'Preset';
    }
}

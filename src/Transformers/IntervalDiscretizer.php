<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Vector;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use InvalidArgumentException;
use RuntimeException;

/**
 * Interval Discretizer
 *
 * This transformer creates an equi-width histogram for each continuous
 * feature column and encodes a discrete category with an automatic bin
 * label (a, b, c...n). The Interval Discretizer is helpful when converting
 * continuous features to categorical features so they can be learned by an
 * estimator that supports categorical features natively.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IntervalDiscretizer implements Stateful
{
    protected const START_CATEGORY = 'a';

    /**
     * The number of bins per feature column.
     *
     * @var int
     */
    protected $bins;

    /**
     * The categories available for each feature column.
     *
     * @var array
     */
    protected $categories;

    /**
     * The bin intervals of the fitted data.
     *
     * @var array[]|null
     */
    protected $intervals;

    /**
     * @param int $bins
     * @throws \InvalidArgumentException
     */
    public function __construct(int $bins = 5)
    {
        if ($bins < 2) {
            throw new InvalidArgumentException('Must have at least 2'
                . " bins per feature column, $bins given.");
        }

        $last = chr(ord(self::START_CATEGORY) + ($bins - 1));

        $this->bins = $bins;
        $this->categories = range(self::START_CATEGORY, $last);
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->intervals);
    }

    /**
     * Return the possible categories of each feature column.
     *
     * @return array
     */
    public function categories() : array
    {
        return $this->categories;
    }

    /**
     * Return the intervals of each continuous feature column
     * calculated during fitting.
     *
     * @return array|null
     */
    public function intervals() : ?array
    {
        return $this->intervals;
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        $this->intervals = [];

        $columns = $dataset->columnsByType(DataType::CONTINUOUS);

        foreach ($columns as $column => $values) {
            $min = min($values);
            $max = max($values);

            $edges = Vector::linspace($min, $max, $this->bins + 1)->asArray();

            array_shift($edges);

            $this->intervals[$column] = $edges;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->intervals === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }
        
        $last = end($this->categories);

        foreach ($samples as &$sample) {
            foreach ($this->intervals as $column => $interval) {
                $value = &$sample[$column];

                foreach ($interval as $i => $edge) {
                    if ($value < $edge) {
                        $value = $this->categories[$i];

                        continue 2;
                    }
                }

                $value = $last;
            }
        }
    }
}

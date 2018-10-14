<?php

namespace Rubix\ML\Transformers;

use Rubix\Tensor\Vector;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use RuntimeException;

/**
 * Interval Discretizer
 *
 * This transformer creates an equi-width histogram for each continuous
 * feature column and encodes a discrete category with an automatic bin
 * label. The Interval Discretizer is helpful when converting continuous
 * features to categorical features so they can be learned by an
 * estimator that supports categorical features natively.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IntervalDiscretizer implements Transformer, Stateful
{
    const START_CATEGORY = 'a';

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
     * @param  int  $bins
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $bins = 5)
    {
        if ($bins < 2) {
            throw new InvalidArgumentException('Must have at least 2'
                . ' bins per feature column.');
        }

        $last = chr(ord(self::START_CATEGORY) + ($bins - 1));

        $this->bins = $bins;
        $this->categories = range(self::START_CATEGORY, $last);
    }

    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->intervals = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataFrame::CONTINUOUS) {
                $values = $dataset->column($column);

                list($min, $max) = Stats::range($values);

                $edges = Vector::linspace($min, $max, $this->bins + 1)
                    ->asArray();

                array_shift($edges);

                $this->intervals[$column] = $edges;
            }
        }
    }

    /**
     * Apply the transformation to the sample matrix.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->intervals)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->intervals as $column => $interval) {
                $category = end($this->categories);

                foreach ($interval as $i => $edge) {
                    if ($sample[$column] < $edge) {
                        $category = $this->categories[$i];

                        break 1;
                    }
                }

                $sample[$column] = $category;
            }
        }
    }
}

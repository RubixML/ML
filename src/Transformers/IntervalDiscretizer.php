<?php

namespace Rubix\ML\Transformers;

use Tensor\Vector;
use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function chr;
use function ord;
use function is_null;

/**
 * Interval Discretizer
 *
 * Assigns each continuous feature to a discrete category using equi-width histograms. Useful
 * for converting continuous data to categorical.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IntervalDiscretizer implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The value of the starting category for each feature column.
     *
     * @var string
     */
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
     * @var string[]
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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $bins = 5)
    {
        if ($bins < 3) {
            throw new InvalidArgumentException('Number of bins must be'
                . " greater than 3, $bins given.");
        }

        $last = chr(ord(self::START_CATEGORY) + $bins - 1);

        $categories = array_map('strval', range(self::START_CATEGORY, $last));

        $this->bins = $bins;
        $this->categories = $categories;
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
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->intervals);
    }

    /**
     * Return the list of possible category values for each discretized feature column.
     *
     * @return string[]
     */
    public function categories() : array
    {
        return $this->categories;
    }

    /**
     * Return the intervals for each continuous feature column calculated during fitting.
     *
     * @return array[]|null
     */
    public function intervals() : ?array
    {
        return $this->intervals;
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->intervals = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->column($column);

                $edges = Vector::linspace(min($values), max($values), $this->bins - 1)->asArray();

                $edges[] = INF;

                $this->intervals[$column] = $edges;
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->intervals)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->intervals as $column => $interval) {
                $value = &$sample[$column];

                foreach ($interval as $k => $edge) {
                    if ($value <= $edge) {
                        $value = $this->categories[$k];

                        continue 2;
                    }
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
        return "Interval Discretizer (bins: {$this->bins})";
    }
}

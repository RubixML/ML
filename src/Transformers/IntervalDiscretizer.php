<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\linspace;
use function array_slice;

/**
 * Interval Discretizer
 *
 * Assigns continuous features to ordered categories using variable width per-feature histograms with a fixed
 * user-specified number of bins.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class IntervalDiscretizer implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The number of bins per histogram.
     *
     * @var int
     */
    protected int $bins;

    /**
     * The bin intervals of the fitted data.
     *
     * @var array[]|null
     */
    protected ?array $intervals = null;

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

        $this->bins = $bins;
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
     * Return the bin intervals of the fitted data.
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

        $q = linspace(0.0, 1.0, 1 + $this->bins);

        $q = array_slice($q, 1, -1);

        $this->intervals = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isContinuous()) {
                $values = $dataset->feature($column);

                $edges = Stats::quantiles($values, $q);

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
        if ($this->intervals === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->intervals as $column => $interval) {
                $value = &$sample[$column];

                foreach ($interval as $ordinal => $edge) {
                    if ($value <= $edge) {
                        $value = "$ordinal";

                        break;
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

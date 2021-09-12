<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function array_merge;
use function array_fill;

/**
 * Target Encoder
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TargetEncoder implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The expected values of each target conditioned on a categorical feature.
     *
     * @var array[]|null
     */
    protected ?array $means = null;

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
        return isset($this->categories);
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Transformer requires a'
                . ' Labeled training set.');
        }

        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $labels = $dataset->labels();
        $labelTypeCode = $dataset->labelType()->code();

        $this->means = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if ($type->isCategorical()) {
                $values = $dataset->feature($column);

                $strata = [];

                foreach ($values as $i => $category) {
                    $strata[$category][] = $labels[$i];
                }

                switch ($labelTypeCode) {
                    case DataType::CONTINUOUS:
                        $means = [];

                        foreach ($strata as $category => $stratum) {
                            $means[$category] = Stats::mean($stratum);
                        }

                        $this->means[$column] = $means;
                }
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
        if ($this->categories === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->categories as $column => $categories) {
                $category = $sample[$column];

                $template = array_fill(0, count($categories), 0);

                if (isset($categories[$category])) {
                    $template[$categories[$category]] = 1;
                }

                $vectors[] = $template;

                unset($sample[$column]);
            }

            $sample = array_merge($sample, ...$vectors);
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
        return 'One Hot Encoder';
    }
}

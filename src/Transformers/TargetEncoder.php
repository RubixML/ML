<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function array_keys;
use function array_merge;
use function array_fill_keys;
use function array_count_values;

/**
 * Target Encoder
 *
 * The Target Encoder converts categorical feature columns into continuous representations with
 * statistics derived from the label or *target* of a supervised dataset. For a continuous target,
 * each category is replaced by the Bayesian-smoothed mean of the target over that category. For
 * a categorical target, each category is replaced by the Bayesian-smoothed probability of each
 * class, thereby expanding the single column into one column per class. Smoothing pulls the
 * per-category estimate toward the global estimate, which reduces the chance that rare or
 * low-frequency categories are given disproportionately large influence in the fit. Categories
 * not observed during fitting are encoded with the global prior.
 *
 * > **Note:** This transformer is *supervised* and must be fitted to a Labeled dataset.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TargetEncoder implements Transformer, Stateful, Persistable
{
    use AutotrackRevisions;

    /**
     * The amount of Bayesian smoothing applied to each per-category estimate.
     *
     * @var float
     */
    protected float $smoothing;

    /**
     * Whether the fitted target was of categorical data type.
     *
     * @var bool
     */
    protected bool $isCategoricalLabel;

    /**
     * The per-category encodings indexed by feature column. For a categorical target each
     * value is a list of K class probabilities; for a continuous target each value is a scalar.
     *
     * @var array<int, array<string|int, float|list<float>>>|null
     */
    protected ?array $encodings = null;

    /**
     * The ordered list of classes computed during fitting, or null if the target is continuous.
     *
     * @var array<string|int>|null
     */
    protected ?array $classOrder = null;

    /**
     * The global (Laplace-smoothed) class priors used as a fallback for unseen categories.
     *
     * @var list<float>|null
     */
    protected ?array $globalPriors = null;

    /**
     * The global mean of the continuous target used as a fallback for unseen categories.
     *
     * @var float|null
     */
    protected ?float $globalMean = null;

    /**
     * Build a new target encoder.
     *
     * @param float $smoothing
     * @throws InvalidArgumentException
     */
    public function __construct(float $smoothing = 1.0)
    {
        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        $this->smoothing = $smoothing;
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
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->encodings);
    }

    /**
     * Return the per-category encodings computed during fitting, indexed by feature column.
     *
     * @return array<int, array<string|int, float|list<float>>>|null
     */
    public function encodings() : ?array
    {
        return $this->encodings;
    }

    /**
     * Return the global prior used as a fallback for categories not observed during fitting.
     *
     * @return list<float>|null
     */
    public function globalPrior() : ?array
    {
        if ($this->isCategoricalLabel) {
            return $this->globalPriors;
        }

        return $this->globalMean !== null ? [$this->globalMean] : null;
    }

    /**
     * Fit the transformer to a labeled dataset.
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Transformer requires a'
                . ' Labeled training set.');
        }

        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $this->encodings = null;
        $this->classOrder = null;
        $this->globalPriors = null;
        $this->globalMean = null;
        $this->isCategoricalLabel = false;

        $m = $this->smoothing;

        $isCategorical = $dataset->labelType()->isCategorical();

        $samples = $dataset->samples();
        $labels = $dataset->labels();

        $n = count($samples);

        $labelsByClass = $isCategorical ? array_count_values($labels) : null;

        $globalMean = $isCategorical ? 0.0 : Stats::mean($labels);

        $numClasses = $isCategorical ? max(1, count($labelsByClass)) : 0;

        $classOrder = $isCategorical ? array_keys($labelsByClass) : null;

        $globalPriors = null;

        if ($isCategorical) {
            $total = (float) $n + $m * $numClasses;

            $globalPriors = [];

            foreach ($classOrder as $label) {
                $globalPriors[] = ($labelsByClass[$label] + $m) / $total;
            }
        }

        $encodings = [];

        foreach ($dataset->featureTypes() as $column => $type) {
            if (!$type->isCategorical()) {
                continue;
            }

            $catCounts = null;
            $catSums = null;
            $catFreqs = null;

            foreach ($samples as $i => $sample) {
                $category = $sample[$column];

                if ($isCategorical) {
                    if (empty($catCounts[$category])) {
                        $catCounts[$category] = array_fill_keys($classOrder, 0);
                        $catFreqs[$category] = 0;
                    }

                    ++$catCounts[$category][$labels[$i]];
                    ++$catFreqs[$category];
                } else {
                    if (!isset($catSums[$category])) {
                        $catSums[$category] = 0.0;
                        $catFreqs[$category] = 0;
                    }

                    $catSums[$category] += $labels[$i];
                    ++$catFreqs[$category];
                }
            }

            $columnEncodings = [];

            if ($isCategorical) {
                foreach ($catCounts as $category => $counts) {
                    $freq = $catFreqs[$category];
                    $denom = $freq + $m;

                    $encoded = [];

                    foreach ($classOrder as $j => $label) {
                        $encoded[] = ($counts[$label] + $m * $globalPriors[$j]) / $denom;
                    }

                    $columnEncodings[$category] = $encoded;
                }
            } else {
                foreach ($catSums as $category => $sum) {
                    $freq = $catFreqs[$category];

                    $columnEncodings[$category] = ($sum + $m * $globalMean) / ($freq + $m);
                }
            }

            $encodings[$column] = $columnEncodings;
        }

        $this->encodings = $encodings;

        $this->isCategoricalLabel = $isCategorical;
        $this->classOrder = $classOrder;
        $this->globalPriors = $globalPriors;
        $this->globalMean = $isCategorical ? null : $globalMean;
    }

    /**
     * Transform the dataset in place.
     *
     * @param list<list<mixed>> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if ($this->encodings === null) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        if ($this->isCategoricalLabel) {
            $this->transformCategorical($samples);

            return;
        }

        $this->transformContinuous($samples);
    }

    /**
     * Replace each categorical category with a list of smoothed per-class probabilities.
     *
     * @param list<list<mixed>> $samples
     */
    protected function transformCategorical(array &$samples) : void
    {
        $fallback = $this->globalPriors;

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->encodings as $column => $categories) {
                $category = $sample[$column];

                $vectors[] = $categories[$category]
                    ?? $fallback;

                unset($sample[$column]);
            }

            $sample = array_merge($sample, ...$vectors);
        }

        unset($sample);
    }

    /**
     * Replace each categorical category in place with its smoothed target mean.
     *
     * @param list<list<mixed>> $samples
     */
    protected function transformContinuous(array &$samples) : void
    {
        $fallback = $this->globalMean;

        foreach ($samples as &$sample) {
            foreach ($this->encodings as $column => $categories) {
                $category = $sample[$column];

                $sample[$column] = $categories[$category]
                    ?? $fallback;
            }
        }

        unset($sample);
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
        return "Target Encoder (smoothing: {$this->smoothing})";
    }
}

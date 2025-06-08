<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Local Outlier Factor
 *
 * Local Outlier Factor (LOF) measures the local deviation of density of an unknown
 * sample with respect to its *k* nearest neighbors from the training set. As such,
 * LOF only considers the local region (or *neighborhood*) of an unknown sample
 * which enables it to detect anomalies within individual clusters of data.
 *
 * References:
 * [1] M. M. Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LocalOutlierFactor implements Estimator, Learner, Scoring, Persistable
{
    use AutotrackRevisions;

    protected const DEFAULT_THRESHOLD = 1.5;

    protected int $k;
    protected ?float $contamination;
    protected Spatial $tree;
    protected array $kdistances = [];
    protected array $lrds = [];
    protected ?float $threshold = null;
    protected ?int $featureCount = null;

    /**
     * @param int $k The number of nearest neighbors to consider
     * @param float|null $contamination The expected proportion of outliers
     * @param Spatial|null $tree The spatial tree to use for nearest neighbor searches
     * 
     * @throws InvalidArgumentException
     */
    public function __construct(int $k = 20, ?float $contamination = null, ?Spatial $tree = null)
    {
        if ($k < 1) {
            throw new InvalidArgumentException("At least 1 neighbor is required, $k given.");
        }

        if ($contamination !== null && ($contamination < 0.0 || $contamination > 0.5)) {
            throw new InvalidArgumentException("Contamination must be between 0 and 0.5, $contamination given.");
        }

        $this->k = $k;
        $this->contamination = $contamination;
        $this->tree = $tree ?? new KDTree();
    }

    public function type(): EstimatorType
    {
        return EstimatorType::anomalyDetector();
    }

    public function compatibility(): array
    {
        return $this->tree->kernel()->compatibility();
    }

    public function params(): array
    {
        return [
            'k' => $this->k,
            'contamination' => $this->contamination,
            'tree' => $this->tree,
        ];
    }

    public function trained(): bool
    {
        return !$this->tree->bare() && $this->kdistances && $this->lrds;
    }

    public function tree(): Spatial
    {
        return $this->tree;
    }

    public function train(Dataset $dataset): void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $labels = range(0, $dataset->numSamples() - 1);
        $dataset = Labeled::quick($dataset->samples(), $labels);

        $this->tree->grow($dataset);
        $this->kdistances = $this->lrds = [];

        $indicesList = $distancesList = [];
        foreach ($dataset->samples() as $sample) {
            [,, $distances] = $this->tree->nearest($sample, $this->k);
            $this->kdistances[] = end($distances) ?: INF;
            $indicesList[] = array_keys($distances);
            $distancesList[] = array_values($distances);
        }

        $this->lrds = array_map(
            fn(array $indices, array $distances) => $this->localReachabilityDensity($indices, $distances),
            $indicesList,
            $distancesList
        );

        if ($this->contamination !== null) {
            $lofs = array_map([$this, 'localOutlierFactor'], $dataset->samples());
            $this->threshold = Stats::quantile($lofs, 1.0 - $this->contamination);
        } else {
            $this->threshold = self::DEFAULT_THRESHOLD;
        }

        $this->featureCount = $dataset->numFeatures();
    }

    public function predict(Dataset $dataset): array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    public function predictSample(array $sample): int
    {
        return $this->scoreSample($sample) > $this->threshold ? 1 : 0;
    }

    public function score(Dataset $dataset): array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        return array_map([$this, 'scoreSample'], $dataset->samples());
    }

    protected function scoreSample(array $sample): float
    {
        [,$indices, $distances] = $this->tree->nearest($sample, $this->k);
        $lrd = $this->localReachabilityDensity($indices, $distances) ?: EPSILON;

        $ratios = array_map(
            fn($index) => $this->lrds[$index] / $lrd,
            $indices
        );

        return Stats::mean($ratios);
    }

    protected function localReachabilityDensity(array $indices, array $distances): float
    {
        $rds = array_map(
            fn($distance, $index) => max($distance, $this->kdistances[$index]),
            $distances,
            $indices
        );

        return 1.0 / (Stats::mean($rds) ?: EPSILON);
    }

    public function __toString(): string
    {
        return 'Local Outlier Factor (' . Params::stringify($this->params()) . ')';
    }
}

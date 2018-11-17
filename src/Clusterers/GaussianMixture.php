<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gaussian Mixture
 *
 * A Gaussian Mixture model is a probabilistic model for representing the
 * presence of clusters within an overall population without requiring a sample
 * to know which sub-population it belongs to a priori. GMMs are similar to
 * centroid-based clusterers like K Means but allow not just the means to
 * be learned but the variances (or *radii*) as well.
 *
 * References:
 * [1] A. P. Dempster et al. (1977). Maximum Likelihood from Incomplete Data via
 * the EM Algorithm.
 * [2] J. Blomer et al. (2016). Simple Methods for Initializing the EM Algorithm
 * for Gaussian Mixture Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianMixture implements Learner, Probabilistic, Verbose, Persistable
{
    use LoggerAware;
    
    const TWO_PI = 2. * M_PI;

    /**
     * The number of gaussian components to fit to the training set i.e. the
     * target number of clusters.
     *
     * @var int
     */
    protected $k;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The minimum change in the components necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The precomputed prior probabilities of each cluster given by weight.
     *
     * @var array
     */
    protected $priors = [
        //
    ];

    /**
     * The computed means of each feature column for each gaussian.
     *
     * @var array
     */
    protected $means = [
        //
    ];

    /**
     * The computed variances of each feature column for each gaussian.
     *
     * @var array
     */
    protected $variances = [
        //
    ];

    /**
     * The amount of gaussian shift during each epoch of training.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  int  $k
     * @param  int  $epochs
     * @param  float  $minChange
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, int $epochs = 100, float $minChange = 1e-3)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . " cluster, $k given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . " least 1 epoch, $epochs given.");
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . " than 0, $minChange given.");
        }

        $this->k = $k;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the cluster prior probabilities i.e. the mixing component.
     *
     * @return array
     */
    public function priors() : array
    {
        return $this->priors;
    }

    /**
     * Return the computed mean vectors of each component.
     *
     * @return array
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the multivariate variance of each component.
     *
     * @return array
     */
    public function variances() : array
    {
        return $this->variances;
    }

    /**
     * Return the amount of gaussian shift at each epoch of training.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($this->logger) $this->logger->info('Learner initialized w/ '
            . Params::stringify([
                'k' => $this->k,
                'epochs' => $this->epochs,
                'min_change' => $this->minChange,
            ]));

        $n = $dataset->numRows();

        if ($this->logger) $this->logger->info("Initializing $this->k"
            . ' gaussian components');

        list($means, $variances) = $this->initializeComponents($dataset);

        $this->means = $prevMeans = $means;
        $this->variances = $prevVariances = $variances;

        $this->priors = array_fill(0, $this->k, 1. / $this->k);

        $this->steps = $memberships = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($dataset as $i => $sample) {
                $memberships[$i] = $this->jointLikelihood($sample);
            }

            foreach ($this->priors as $cluster => &$prior) {
                $prior = array_sum(array_column($memberships, $cluster)) / $n;

                $means = $this->means[$cluster];
                $variances = $this->variances[$cluster];

                foreach ($means as $column => $mean) {
                    $a = $b = $total = 0.;

                    foreach ($dataset as $i => $sample) {
                        $prob = $memberships[$i][$cluster];

                        $a += $prob * $sample[$column];
                        $total += $prob;
                    }

                    $means[$column] = $a / ($total ?: self::EPSILON);

                    foreach ($dataset as $i => $sample) {
                        $prob = $memberships[$i][$cluster];

                        $b += $prob * ($sample[$column] - $mean) ** 2;
                    }

                    $variances[$column] = $b / ($total ?: self::EPSILON);
                }

                $this->means[$cluster] = $means;
                $this->variances[$cluster] = $variances;
            }

            $shift = $this->gaussianShift($prevMeans, $prevVariances);

            $this->steps[] = $shift;

            if ($this->logger) $this->logger->info("Epoch $epoch"
                . " complete, shift=$shift");

            if ($shift < $this->minChange) {
                break 1;
            }

            $prevMeans = $this->means;
            $prevVariances = $this->variances;
        }

        if ($this->logger) $this->logger->info('Training complete');
    }

    /**
     * Make predictions from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([Argmax::class, 'compute'], $this->proba($dataset));
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if (empty($this->priors)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([self::class, 'jointLikelihood'], $dataset->samples());
    }

    /**
     * Initialize the gaussian components using K Means.
     * 
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    protected function initializeComponents(Dataset $dataset) : array
    {
        $clusterer = new KMeans($this->k);

        $clusterer->train($dataset);

        $labels = $clusterer->predict($dataset);

        $bootstrap = Labeled::quick($dataset->samples(), $labels);

        $means = $variances = [];

        foreach ($bootstrap->stratify() as $cluster => $stratum) {
            foreach ($stratum->columns() as $column => $values) {
                list($mean, $variance) = Stats::meanVar($values);

                $means[$cluster][] = $mean;
                $variances[$cluster][] = $variance;
            }
        }

        return [$means, $variances];
    }

    /**
     * Calculate the normalized joint likelihood of a sample being a member
     * of each of the gaussian components.
     *
     * @param  array  $sample
     * @return array
     */
    protected function jointLikelihood(array $sample) : array
    {
        $likelihood = [];

        foreach ($this->priors as $cluster => $prior) {
            $means = $this->means[$cluster];
            $variances = $this->variances[$cluster];

            $score = $prior;

            foreach ($sample as $column => $feature) {
                $mean = $means[$column];
                $variance = $variances[$column];

                $pdf = 1. / sqrt(self::TWO_PI * $variance);
                $pdf *= exp(-(($feature - $mean) ** 2 / (2. * $variance)));

                $score *= $pdf;
            }

            $likelihood[$cluster] = $score;
        }

        $total = array_sum($likelihood) ?: self::EPSILON;

        foreach ($likelihood as &$probability) {
            $probability /= $total;
        }

        return $likelihood;
    }

    /**
     * Calculate the magnitude (l2) of gaussian shift from the previous epoch.
     *
     * @param  array  $prevMeans
     * @param  array  $prevVariances
     * @return float
     */
    protected function gaussianShift(array $prevMeans, array $prevVariances) : float
    {
        $shift = 0.;

        foreach ($this->means as $cluster => $means) {
            $variances = $this->variances[$cluster];

            $prevMean = $prevMeans[$cluster];
            $prevVariance = $prevVariances[$cluster];

            foreach ($means as $column => $mean) {
                $variance = $variances[$column];

                $shift += ($prevMean[$column] - $mean) ** 2;
                $shift += ($prevVariance[$column] - $variance) ** 2;
            }
        }

        return $shift;
    }
}

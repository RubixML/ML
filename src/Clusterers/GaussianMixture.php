<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Functions\Argmax;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Functions\LogSumExp;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gaussian Mixture
 *
 * A Gaussian Mixture model (GMM) is a probabilistic model for representing the
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
     * @var float[]
     */
    protected $priors = [
        //
    ];

    /**
     * The computed means of each feature column for each gaussian.
     *
     * @var array[]
     */
    protected $means = [
        //
    ];

    /**
     * The computed variances of each feature column for each gaussian.
     *
     * @var array[]
     */
    protected $variances = [
        //
    ];

    /**
     * The amount of gaussian shift during each epoch of training.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param int $k
     * @param int $epochs
     * @param float $minChange
     * @throws \InvalidArgumentException
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
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::CLUSTERER;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->means and $this->variances;
    }

    /**
     * Return the cluster prior probabilities.
     *
     * @return float[]
     */
    public function priors() : array
    {
        $priors = [];

        if (is_array($this->priors)) {
            $total = LogSumExp::compute($this->priors);

            foreach ($this->priors as $class => $probability) {
                $priors[$class] = exp($probability - $total);
            }
        }

        return $priors;
    }

    /**
     * Return the computed mean vectors of each component.
     *
     * @return array[]
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the multivariate variance of each component.
     *
     * @return array[]
     */
    public function variances() : array
    {
        return $this->variances;
    }

    /**
     * Return the loss at each epoch of training.
     *
     * @return float[]
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        if ($this->logger) {
            $this->logger->info('Learner initialized w/ '
                . Params::stringify([
                    'k' => $this->k,
                    'epochs' => $this->epochs,
                    'min_change' => $this->minChange,
                ]));
        }

        $n = $dataset->numRows();

        $columns = $dataset->columns();

        if ($this->logger) {
            $this->logger->info("Initializing $this->k"
                . ' gaussian components');
        }

        [$means, $variances] = $this->initializeComponents($dataset);

        $this->means = $means;
        $this->variances = $variances;

        $this->priors = array_fill(0, $this->k, log(1. / $this->k));

        $this->steps = [];

        $prevLoss = 0.;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $memberships = [];

            foreach ($dataset as $sample) {
                $jll = $this->jointLogLikelihood($sample);

                $memberships[] = array_map('exp', $jll);
            }

            $loss = 0.;

            for ($cluster = 0; $cluster < $this->k; $cluster++) {
                $mHat = array_column($memberships, $cluster);

                $means = $variances = [];

                foreach ($columns as $values) {
                    $a = $b = $total = 0.;

                    foreach ($values as $i => $value) {
                        $membership = $mHat[$i];

                        $a += $membership * $value;
                        $total += $membership;
                    }

                    $total = $total ?: self::EPSILON;

                    $mean = $a / $total;

                    foreach ($values as $i => $value) {
                        $b += $mHat[$i] * ($value - $mean) ** 2;
                    }

                    $variance = $b / $total;

                    $means[] = $mean;
                    $variances[] = $variance ?: self::EPSILON;

                    $loss += $total;
                }

                $prior = array_sum($mHat) / $n;

                $this->means[$cluster] = $means;
                $this->variances[$cluster] = $variances;
                $this->priors[$cluster] = log($prior);
            }

            $this->steps[] = $loss;

            if ($this->logger) {
                $this->logger->info("Epoch $epoch complete, loss=$loss");
            }

            if (is_nan($loss)) {
                break 1;
            }

            if (abs($loss - $prevLoss) < $this->minChange) {
                break 1;
            }

            $prevLoss = $loss;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @throws \InvalidArgumentException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (empty($this->priors)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $predictions = [];

        foreach ($dataset as $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $predictions[] = Argmax::compute($jll);
        }

        return $predictions;
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->priors)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $probabilities = [];

        foreach ($dataset as $sample) {
            $jll = $this->jointLogLikelihood($sample);

            $total = LogSumExp::compute($jll);

            $dist = [];

            foreach ($jll as $cluster => $likelihood) {
                $dist[$cluster] = exp($likelihood - $total);
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }

    /**
     * Initialize the gaussian components using K Means.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    protected function initializeComponents(Dataset $dataset) : array
    {
        $clusterer = new KMeans($this->k);

        $clusterer->train($dataset);

        $labels = $clusterer->predict($dataset);

        $dataset = Labeled::quick($dataset->samples(), $labels);

        $means = $variances = [];

        foreach ($dataset->stratify() as $cluster => $stratum) {
            $mHat = $vHat = [];

            foreach ($stratum->columns() as $values) {
                [$mean, $variance] = Stats::meanVar($values);

                $mHat[] = $mean;
                $vHat[] = $variance;
            }

            $means[$cluster] = $mHat;
            $variances[$cluster] = $vHat;
        }

        return [$means, $variances];
    }

    /**
     * Calculate the joint log likelihood of a sample being a member
     * of each of the gaussian components.
     *
     * @param array $sample
     * @return array
     */
    protected function jointLogLikelihood(array $sample) : array
    {
        $likelihoods = [];

        foreach ($this->priors as $cluster => $prior) {
            $means = $this->means[$cluster];
            $variances = $this->variances[$cluster];

            $likelihood = $prior;

            foreach ($sample as $column => $feature) {
                $mean = $means[$column];
                $variance = $variances[$column];

                $pdf = -0.5 * log(self::TWO_PI * $variance);
                $pdf -= 0.5 * (($feature - $mean) ** 2) / $variance;

                $likelihood += $pdf;
            }

            $likelihoods[$cluster] = $likelihood;
        }

        return $likelihoods;
    }
}

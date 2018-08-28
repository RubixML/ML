<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use Rubix\ML\Other\Functions\Argmax;
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
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianMixture implements Estimator, Probabilistic, Persistable
{
    const TWO_PI = 2. * M_PI;

    /**
     * The number of gaussian components to fit to the training set i.e. the
     * target number of clusters.
     *
     * @var int
     */
    protected $k;

    /**
     * The minimum change in the components necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The precomputed prior log probabilities of each cluster given by weight.
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
     * @param  float  $minChange
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $k, float $minChange = 1e-3, int $epochs = 100)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('Must target at least one'
                . ' cluster.');
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        $this->k = $k;
        $this->minChange = $minChange;
        $this->epochs = $epochs;
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
        if (in_array(Dataset::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $n = $dataset->numRows();

        if ($n < $this->k) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the number of components.');
        }

        $this->priors = array_fill(0, $this->k, 1. / $this->k);

        $this->means = $previous = $dataset->randomize()->tail($this->k)->samples();

        $this->variances = array_fill(0, $this->k, array_fill(0,
            $dataset->numColumns(), 1.));

        $this->steps = $memberships = [];

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($dataset as $i => $sample) {
                $memberships[$i] = $this->jointLikelihood($sample);
            }

            foreach ($this->means as $cluster => &$means) {
                $variances = $this->variances[$cluster];

                foreach ($means as $column => &$mean) {
                    $a = $b = $total = self::EPSILON;

                    foreach ($dataset as $i => $sample) {
                        $prob = $memberships[$i][$cluster];

                        $a += $prob * $sample[$column];
                        $total += $prob;
                    }

                    $mean = $a / $total;

                    foreach ($dataset as $i => $sample) {
                        $prob = $memberships[$i][$cluster];

                        $b += $prob * ($sample[$column] - $mean) ** 2;
                    }

                    $variances[$column] = $b / $total;
                }

                $this->variances[$cluster] = $variances;
            }

            foreach ($this->priors as $cluster => &$prior) {
                $prior = array_sum(array_column($memberships, $cluster)) / $n;
            }

            $shift = $this->calculateGaussianShift($previous);

            $this->steps[] = $shift;

            if ($shift < $this->minChange) {
                break 1;
            }

            $previous = $this->means;
        }
    }

    /**
     * Make a prediction based on the cluster probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($this->proba($dataset) as $probabilities) {
            $predictions[] = Argmax::compute($probabilities);
        }

        return $predictions;
    }

    /**
     * Return an array of cluster probabilities for each sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        if (empty($this->priors)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $probabilities = [];

        foreach ($dataset as $i => $sample) {
            $probabilities[$i] = $this->jointLikelihood($sample);
        }

        return $probabilities;
    }

    /**
     * Calculate the joint log likelihood of a sample being a member of each of
     * the gaussians.
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
                $variance = $variances[$column] + self::EPSILON;

                $pdf = 1. / (self::TWO_PI * $variance) ** 0.5;
                $pdf *= M_E ** (-(($feature - $mean) ** 2 / (2. * $variance)));

                $score *= $pdf;
            }

            $likelihood[$cluster] = $score;
        }

        $total = array_sum($likelihood);

        foreach ($likelihood as &$probability) {
            $probability /= $total;
        }

        return $likelihood;
    }

    /**
     * Calculate the magnitude (l1) of gaussian shift from the previous epoch.
     *
     * @param  array  $previous
     * @return float
     */
    protected function calculateGaussianShift(array $previous) : float
    {
        $shift = 0.;

        foreach ($this->means as $cluster => $means) {
            $prevCluster = $previous[$cluster];

            foreach ($means as $column => $mean) {
                $shift += abs($prevCluster[$column] - $mean);
            }
        }

        return $shift;
    }
}

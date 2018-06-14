<?php

namespace Rubix\ML\Clusterers;

use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Metrics\Distance\Distance;
use Rubix\ML\Metrics\Distance\Euclidean;
use InvalidArgumentException;
use RuntimeException;

class FuzzyCMeans implements Clusterer, Probabilistic, Persistable
{
    /**
     * The target number of clusters.
     *
     * @var int
     */
    protected $c;

    /**
     * This measures the tolerance the clusterer has to overlap and will
     * determine the bandwidth of the fuzzy area. i.e. The fuzz factor.
     *
     * @var float
     */
    protected $fuzz;

    /**
     * The distance function to use when computing the distances between
     * samples.
     *
     * @var \Rubix\ML\Metrics\Distance\Distance
     */
    protected $distanceFunction;

    /**
     * The sensitivity threshold. i.e. the minimum change in the centroid means
     * necessary for the algorithm to continue learning.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The maximum number of iterations to run until the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The computed centroid vectors of the training data.
     *
     * @var int
     */
    protected $centroids = [
        //
    ];

    /**
     * An array holding the progress of the last training session. i.e. the
     * total distance between all the centroids and data points.
     *
     * @var array
     */
    protected $progress = [
        //
    ];

    /**
     * @param  int  $c
     * @param  float  $fuzz
     * @param  \Rubix\ML\Contracts\Distance  $distanceFunction
     * @param  float  $threshold
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $c, float $fuzz = 2.0, Distance $distanceFunction = null,
                            float $threshold = 1e-4, int $epochs = PHP_INT_MAX)
    {
        if ($c < 1) {
            throw new InvalidArgumentException('Target clusters must be'
                . ' greater than 1.');
        }

        if ($fuzz < 1) {
            throw new InvalidArgumentException('Fuzz factor must be greater'
                . ' than 1.');
        }

        if ($threshold < 0) {
            throw new InvalidArgumentException('Threshold cannot be set to less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($distanceFunction)) {
            $distanceFunction = new Euclidean();
        }

        $this->c = $c;
        $this->fuzz = $fuzz + self::EPSILON;
        $this->distanceFunction = $distanceFunction;
        $this->threshold = $threshold;
        $this->epochs = $epochs;
    }

    /**
     * Return the computed cluster centroids of the training data.
     *
     * @return array
     */
    public function centroids() : array
    {
        return $this->centroids;
    }

    /**
     * Return the progress from last training session.
     *
     * @return array
     */
    public function progress() : array
    {
        return $this->progress;
    }

    /**
     * Pick C random samples and assign them as centroids. Compute the coordinates
     * of the centroids by clustering the points based on each sample's distance
     * from one of the C centroids, then recompute the centroid coordinate as the
     * mean of the new cluster.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return array
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        if ($dataset->numRows() < $this->c) {
            throw new RuntimeException('The number of samples cannot be less'
                . ' than the parameter C.');
        }

        $this->centroids = array_fill(0, $this->c,
            array_fill(0, $dataset->numColumns(), 0.0));

        $memberships = $this->initializeMemberships($dataset->numRows());

        $this->step($dataset, $memberships);

        $this->progress = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($dataset as $index => $sample) {
                $memberships[$index] = $this->calculateMembership($sample);
            }

            $this->step($dataset, $memberships);

            $score = $this->scoreEpoch($dataset);

            $this->progress[$epoch] = $score;

            if (count($this->progress) > 2) {
                $last = $this->progress[count($this->progress) - 2];

                if (abs($last - $score) < $this->threshold) {
                    break 1;
                }
            }
        }
    }

    /**
     * Make a prediction based on the cluster probabilities.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $probabilities) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $label => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $label;
                }
            }

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Return an array of cluster probabilities for each sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $probabilities = [];

        foreach ($samples as $sample) {
            $probabilities[] = $this->calculateMembership($sample);
        }

        return $probabilities;
    }

    /**
     * Return an vector of membership probability score of each cluster for a
     * given sample.
     *
     * @param  array  $sample
     * @return array
     */
    protected function calculateMembership(array $sample) : array
    {
        $membership = [];

        foreach ($this->centroids as $label => $centroid1) {
            $a = $this->distanceFunction->compute($sample, $centroid1);

            $total = self::EPSILON;

            foreach ($this->centroids as $centroid2) {
                $b = $this->distanceFunction->compute($sample, $centroid2);

                $total += pow($a / ($b + self::EPSILON),
                    2 / ($this->fuzz - 1));
            }

            $membership[$label] = 1 / $total;
        }

        return $membership;
    }

    /**
     * Update the cluster centroids with a new membership matrix.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @param  array  $memberships
     * @return void
     */
    protected function step(Dataset $dataset, array $memberships) : void
    {
        foreach ($this->centroids as $label => &$centroid) {
            foreach ($centroid as $j => &$mean) {
                $total = self::EPSILON;
                $a = 0.0;

                foreach ($dataset as $k => $sample) {
                    $weight = pow($memberships[$k][$label], $this->fuzz);

                    $a += $weight * $sample[$j];
                    $total += $weight;
                }

                $mean = $a / $total;
            }
        }
    }

    /**
     * Initialize the membership matrix of dimension n x C.
     *
     * @param  int  $n
     * @return array
     */
    protected function initializeMemberships(int $n) : array
    {
        $memberships = array_fill(0, $n, array_fill(0, $this->c, 0.0));

        for ($i = 0; $i < $n; $i++) {
            $total = 0.0;

            for ($j = 0; $j < $this->c; $j++) {
                $weight = random_int(0, 1e8) / 1e8;

                $memberships[$i][$j] = $weight;

                $total += $weight;
            }

            foreach ($memberships[$i] as &$membership) {
                $membership /= $total;
            }
        }

        return $memberships;
    }

    /**
     * Return a score based on the objective function. i.e. the minimum total
     * distance between cluster centroids and all samples.
     *
     * @return float
     */
    protected function scoreEpoch(Dataset $dataset) : float
    {
        $score = 0.0;

        foreach ($dataset as $sample) {
            foreach ($this->centroids as $centroid) {
                $score += $this->distanceFunction->compute($sample, $centroid);
            }
        }

        return $score;
    }
}

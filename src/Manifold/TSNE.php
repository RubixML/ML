<?php

namespace Rubix\ML\Manifold;

use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

/**
 * t-SNE
 *
 * T-distributed Stochastic Neighbor Embedding is a two-stage non-linear
 * manifold learning algorithm based on batch Gradient Decent. During the first
 * stage (*early* stage) the samples are exaggerated to encourage distant
 * clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient,
 * momentum is employed to help escape bad local minima.
 *
 * References:
 * [1] L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
 * [2] L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving
 * Local Structure.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TSNE implements Estimator, Verbose
{
    use LoggerAware;

    const BINARY_PRECISION = 100;
    const SEARCH_TOLERANCE = 1e-5;

    /**
     * The number of dimensions of the embedding.
     *
     * @var int
     */
    protected $dimensions;

     /**
      * The number of degrees of freedom for the student t distribution.
      *
      * @var int
      */
    protected $degrees;

    /**
     * The number of effective nearest neighbors to refer to when computing
     * the variance of the gaussian over that sample.
     * 
     * @var float
     */
    protected $perplexity;

     /**
      * The desired entropy of the Gaussian over each sample i.e the log
      * perplexity.
      *
      * @var float
      */
    protected $entropy;

     /**
      * The factor to exaggerate the distances between samples by during the
      * early stage of fitting.
      *
      * @var float
      */
    protected $exaggeration;

    /**
     * The number of times to iterate over the embedding.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The number of epochs that are considered to be in the early training
     * stage.
     *
     * @var int
     */
    protected $early;

    /**
     * The learning rate.
     *
     * @var float
     */
    protected $rate;

    /**
     * The amount of momentum to carry over into the next update.
     *
     * @var float
     */
    protected $momentum;

    /**
     * The minimum gradient necessary to continue fitting.
     *
     * @var float
     */
    protected $minGradient;

    /**
     * The training window to consider during early stop checking i.e. the last
     * n epochs.
     *
     * @var int
     */
    protected $window;

    /**
     * The distance metric used to measure distances between samples in both
     * high and low dimensions.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The magnitudes of the gradient at each epoch since the last embedding.
     *
     * @var float[]
     */
    protected $steps = [
        //
    ];

    /**
     * @param  int  $dimensions
     * @param  int  $perplexity
     * @param  float  $exaggeration
     * @param  int  $epochs
     * @param  float  $rate
     * @param  float  $momentum
     * @param  float  $minGradient
     * @param  int  $window
     * @param  \Rubix\ML\Kernels\Distance\Distance|null  $kernel
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $dimensions = 2, int $perplexity = 30, float $exaggeration = 12.,
                float $rate = 10., float $momentum = 0.5, int $epochs = 1000, float $minGradient = 1e-7,
                int $window = 5, ?Distance $kernel = null)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot embed into less than 1'
                . " dimension, $dimensions given.");
        }

        if ($perplexity < 1) {
            throw new InvalidArgumentException('Perplexity cannot be less than'
                . " 1, $perplexity given.");
        }

        if ($exaggeration < 1.) {
            throw new InvalidArgumentException('Early exaggeration must be 1 or'
             . " greater, $exaggeration given.");
        }

        if ($epochs < 250) {
            throw new InvalidArgumentException('Must iterate for at least 250'
                . " epochs, $epochs given.");
        }

        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . " than 0, $rate given.");
        }

        if ($momentum < 0. or $momentum > 1.) {
            throw new InvalidArgumentException('Momentum must be between 0 and'
                . " 1, $momentum given.");
        }

        if ($minGradient < 0.) {
            throw new InvalidArgumentException('The minimum magnitude of the'
                . " gradient must be 0 or greater, $minGradient given.");
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->dimensions = $dimensions;
        $this->degrees = max($dimensions - 1, 1);
        $this->perplexity = $perplexity;
        $this->entropy = log($perplexity);
        $this->exaggeration = $exaggeration;
        $this->epochs = $epochs;
        $this->early = (int) min(250, round($epochs / 4)) + 1;
        $this->rate = $rate;
        $this->momentum = $momentum;
        $this->minGradient = $minGradient;
        $this->window = $window;
        $this->kernel = $kernel;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::EMBEDDER;
    }

    /**
     * Return the magnitudes of the gradient at each epoch from the last
     * embedding.
     *
     * @return float[]
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Embed a high dimensional sample matrix into a lower dimensional one.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return array[]
     */
    public function predict(Dataset $dataset) : array
    {
        if ($dataset->typeCount(DataFrame::CONTINUOUS) !== $dataset->numColumns()) {
            throw new InvalidArgumentException('This estimator only works'
                . ' with continuous features.');
        }

        if ($this->logger) $this->logger->info('Embedder initialized w/ '
            . Params::stringify([
                'dimensions' => $this->dimensions,
                'perplexity' => $this->perplexity,
                'exaggeration' => $this->exaggeration,
                'epochs' => $this->epochs,
                'rate' => $this->rate,
                'momentum' => $this->momentum,
                'min_gradient' => $this->minGradient,
                'window' => $this->window,
                'kernel' => $this->kernel,
            ]));

        $n = $dataset->numRows();

        $x = Matrix::build($dataset->samples());

        if ($this->logger) $this->logger->info('Computing high dimensional'
            . ' affinities');

        $distances = $this->pairwiseDistances($x);

        $p = $this->highAffinities($distances)
            ->multiply($this->exaggeration);

        $y = $yHat = Matrix::gaussian($n, $this->dimensions)
            ->multiply(1e-3);

        $velocity = Matrix::zeros($n, $this->dimensions);
        $gains = Matrix::ones($n, $this->dimensions)->asArray();

        $this->steps = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            if ($epoch === $this->early) {
                $p = $p->divide($this->exaggeration);

                if ($this->logger) $this->logger->info('Early exaggeration'
                    . ' stage exhausted');
            }

            $distances = $this->pairwiseDistances($y);

            $q = $this->lowAffinities($distances);

            $gradient = $this->computeGradient($p, $q, $y, $distances);

            $magnitude = $gradient->l2Norm();

            $vHat = $velocity->multiply($gradient);

            foreach ($gains as $i => &$row) {
                foreach ($row as $j => &$value) {
                    $value = $vHat[$i][$j] > 0. ? $value + 0.2 : $value * 0.8;
                }
            }

            $gHat = Matrix::quick($gains);

            $gradient = $gradient->multiply($gHat);

            $velocity = $velocity->multiply($this->momentum)
                ->subtract($gradient->multiply($this->rate));

            $y = $y->subtract($velocity);

            $this->steps[] = $magnitude;

            if ($this->logger) $this->logger->info("Epoch $epoch"
                . " complete, gradient=$magnitude");

            if ($magnitude < $this->minGradient) {
                break 1;
            }

            if ($epoch > $this->window) {
                $window = array_slice($this->steps, -$this->window);

                $worst = $window;
                sort($worst);

                if ($window === $worst) {
                    break 1;
                }
            }
        }

        if ($this->logger) $this->logger->info('Embedding complete');

        return $y->asArray();
    }

    /**
     * Calculate the pairwise distances for each sample.
     *
     * @param  \Rubix\Tensor\Matrix  $samples
     * @return \Rubix\Tensor\Matrix
     */
    protected function pairwiseDistances(Matrix $samples) : Matrix
    {
        $distances = [];

        foreach ($samples as $a) {
            $temp = [];

            foreach ($samples as $b) {
                $temp[] = $this->kernel->compute($a, $b);
            }

            $distances[] = $temp;
        }

        return Matrix::quick($distances);
    }

    /**
     * Calculate the joint likelihood of each sample in the high dimensional
     * space as being nearest neighbor to each other sample.
     *
     * @param  \Rubix\Tensor\Matrix  $distances
     * @return \Rubix\Tensor\Matrix
     */
    protected function highAffinities(Matrix $distances) : Matrix
    {
        $p = [[]];

        foreach ($distances as $i => $row) {
            $minBeta = -INF;
            $maxBeta = INF;
            $beta = 1.;

            for ($l = 0; $l < self::BINARY_PRECISION; $l++) {
                $pSigma = 0.;

                foreach ($row as $j => $distance) {
                    if ($i !== $j) {
                        $temp = exp(-$distance * $beta);
                    } else {
                        $temp = 0.;
                    }

                    $p[$i][$j] = $temp;
                    $pSigma += $temp;
                }

                if ($pSigma === 0.) {
                    $pSigma = self::EPSILON;
                }

                $distSigma = 0.;

                foreach ($p[$i] as $j => &$prob) {
                    $prob /= $pSigma;
                    $distSigma += $row[$j] * $prob;
                }

                $entropy = log($pSigma) + $beta * $distSigma;

                $diff = $entropy - $this->entropy;

                if (abs($diff) < self::SEARCH_TOLERANCE) {
                    break 1;
                }

                if ($diff > 0.) {
                    $minBeta = $beta;

                    if ($maxBeta === INF) {
                        $beta *= 2.;
                    } else {
                        $beta = ($beta + $maxBeta) / 2.;
                    }
                } else {
                    $maxBeta = $beta;

                    if ($minBeta === -INF) {
                        $beta /= 2.;
                    } else {
                        $beta = ($beta + $minBeta) / 2.;
                    }
                }
            }
        }

        $p = Matrix::quick($p);

        $pHat = $p->add($p->transpose());

        return $pHat->divide($pHat->sum()->clip(self::EPSILON, INF));
    }

    /**
     * Calculate the joint likelihood of each sample in the embedded space as
     * being nearest neighbor to each other sample.
     *
     * @param  \Rubix\Tensor\Matrix  $distances
     * @return \Rubix\Tensor\Matrix
     */
    protected function lowAffinities(Matrix $distances) : Matrix
    {
        $q = $distances->divide($this->degrees)->add(1.)
            ->pow((1. + $this->degrees) / -2.);

        $qSigma = $q->sum()->clip(self::EPSILON, INF)
            ->multiply(2.);

        return $q->divide($qSigma);
    }

    /**
     * Compute the gradient of the KL Divergence cost function with respect to
     * the embedding.
     *
     * @param  \Rubix\Tensor\Matrix  $p
     * @param  \Rubix\Tensor\Matrix  $q
     * @param  \Rubix\Tensor\Matrix  $y
     * @param  \Rubix\Tensor\Matrix  $distances
     * @return \Rubix\Tensor\Matrix
     */
    protected function computeGradient(Matrix $p, Matrix $q, Matrix $y, Matrix $distances) : Matrix
    {
        $pqd = $p->subtract($q)->multiply($distances);

        $c = 2. * (1. + $this->degrees) / $this->degrees;

        $gradient = [];

        foreach ($pqd->asVectors() as $i => $row) {
            $yHat = $y->rowAsVector($i);
            
            $gradient[] = $row->matmul($y->subtract($yHat))
                ->multiply($c)
                ->row(0);
        }

        return Matrix::quick($gradient);
    }
}

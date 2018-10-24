<?php

namespace Rubix\ML\Manifold;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\DataFrame;
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
class TSNE implements Embedder
{
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
      * The desired entropy of the Gaussian over each sample i.e the log
      * perplexity where perplexity is defined as the number of effective
      * nearest neighbors to refer to when computing the variance of the
      * gaussian over that sample.
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
     * The amount to decay the momentum by each update.
     *
     * @var float
     */
    protected $decay;

    /**
     * The minimum gradient necessary to continue fitting.
     *
     * @var float
     */
    protected $minGradient;

    /**
     * The distance metric used to measure distances between samples in both
     * high and low dimensions.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * The tolerance of the binary search for an appropriate variance of the
     * Gaussian over each sample.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The number of iterations when locating an appropriate variance.
     *
     * @var int
     */
    protected $precision;

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
     * @param  float  $decay
     * @param  float  $minGradient
     * @param  \Rubix\ML\Kernels\Distance\Distance  $kernel
     * @param  float  $tolerance
     * @param  int  $precision
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $dimensions = 2, int $perplexity = 30, float $exaggeration = 12.,
                int $epochs = 1000, float $rate = 1.0, float $decay = 0.2, float $minGradient = 1e-6,
                Distance $kernel = null, float $tolerance = 1e-5, int $precision = 100)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot embed less than 1'
                . ' dimension.');
        }

        if ($perplexity < 1) {
            throw new InvalidArgumentException('Perplexity cannot be less than'
                . ' 1.');
        }

        if ($exaggeration < 1.) {
            throw new InvalidArgumentException('Early exaggeration must be 1 or'
             . ' greater.');
        }

        if ($epochs < 250) {
            throw new InvalidArgumentException('Must iterate for at least 250'
                . ' epochs.');
        }

        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must be greater'
                . ' than 0.');
        }

        if ($decay < 0. or $decay > 1.) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . ' 0 and 1.');
        }

        if ($minGradient < 0.) {
            throw new InvalidArgumentException('The minimum magnitude of the'
                . ' gradient must be 0 or greater.');
        }

        if ($tolerance < 0.) {
            throw new InvalidArgumentException('Binary seach tolerance cannot'
                . ' be less than 0.');
        }

        if ($precision < 1) {
            throw new InvalidArgumentException('Binary search precision must'
                . ' be at at least 1.');
        }

        if (is_null($kernel)) {
            $kernel = new Euclidean();
        }

        $this->dimensions = $dimensions;
        $this->degrees = max($dimensions - 1, 1);
        $this->entropy = log($perplexity);
        $this->exaggeration = $exaggeration;
        $this->epochs = $epochs;
        $this->early = (int) min(250, round($epochs / 4));
        $this->rate = $rate;
        $this->decay = $decay;
        $this->minGradient = $minGradient;
        $this->kernel = $kernel;
        $this->tolerance = $tolerance;
        $this->precision = $precision;
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
    public function embed(Dataset $dataset) : array
    {
        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This embedder only works with'
                . ' continuous features.');
        }

        $this->steps = [];

        $x = Matrix::quick($dataset->samples());

        $y = $yHat = Matrix::gaussian($dataset->numRows(), $this->dimensions)
            ->multiply(1e-3);

        $velocity = Matrix::zeros($dataset->numRows(), $this->dimensions);

        $distances = $this->pairwiseDistances($x);

        $p = $this->highAffinities($distances)
            ->multiply($this->exaggeration);

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            if ($epoch === $this->early) {
                $p = $p->divide($this->exaggeration);
            }

            $distances = $this->pairwiseDistances($y);

            $q = $this->lowAffinities($distances);

            $gradient = $this->computeGradient($p, $q, $y, $distances);

            $velocity = $gradient
                ->multiply($this->rate)
                ->add($velocity->multiply(1. - $this->decay));

            $y = $y->add($velocity);

            $magnitude = $gradient->l2Norm();

            $this->steps[] = $magnitude;

            if ($magnitude < $this->minGradient) {
                break 1;
            }
        }

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

        foreach ($samples as $i => $a) {
            foreach ($samples as $j => $b) {
                $distances[$i][$j] = $this->kernel->compute($a, $b);
            }
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

            for ($l = 0; $l < $this->precision; $l++) {
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

                if (abs($diff) <= $this->tolerance) {
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

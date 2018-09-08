<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Other\Structures\DataFrame;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use InvalidArgumentException;
use RuntimeException;

/**
 * Adaline
 *
 * Adaptive Linear Neuron is a type of single layer neural network with a
 * linear output layer. The Adaline converges to the least squares error
 * which is the same error function as linear regression.
 *
 * References:
 * [1] B. Widrow. (1960). An Adaptive "Adaline" Neuron Using Chemical
 * "Memistors".
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Adaline implements Estimator, Online, Persistable
{
    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The number of training samples to consider per iteration of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The function that computes the cost of an erroneous activation during
     * training.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFunction;

    /**
     * The minimum change in the weights necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The underlying neural network instance.
     *
     * @var \Rubix\ML\NeuralNet\FeedForward|null
     */
    protected $network;

    /**
     * The average cost of a training sample at each epoch.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @param  float  $alpha
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction  $costFunction
     * @param  float  $minChange
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $epochs = 100, int $batchSize = 10, Optimizer $optimizer = null,
                    float $alpha = 1e-4, CostFunction $costFunction = null, float $minChange = 1e-4)
    {
        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Cannot have less than 1 sample'
                . ' per batch.');
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization term must'
                . ' be non-negative.');
        }

        if ($minChange < 0.) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . ' than 0.');
        }

        if (is_null($optimizer)) {
            $optimizer = new Adam();
        }

        if (is_null($costFunction)) {
            $costFunction = new LeastSquares();
        }

        $this->epochs = $epochs;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->costFunction = $costFunction;
        $this->minChange = $minChange;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return self::REGRESSOR;
    }

    /**
     * Return the average cost at every epoch.
     *
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
    }

    /**
     * Return the underlying neural network instance or null if not trained.
     *
     * @return \Rubix\ML\NeuralNet\FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
    * @param  \Rubix\ML\Datasets\Dataset  $dataset
    * @throws \InvalidArgumentException
    * @return void
    */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        $this->network = new FeedForward(
            new Placeholder($dataset->numColumns()),
            [],
            new Continuous($this->alpha),
            $this->costFunction,
            $this->optimizer
        );

        $this->steps = [];

        $this->partial($dataset);
    }

    /**
     * Perform mini-batch gradient descent with given optimizer over the training
     * set and update the input weights accordingly.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('This Estimator requires a'
                . ' Labeled training set.');
        }

        if (in_array(DataFrame::CATEGORICAL, $dataset->types())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        $n = $dataset->numRows();

        if (is_null($this->network)) {
            $this->train($dataset);
        } else {
            $previous = INF;

            for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
                $batches = $dataset->randomize()->batch($this->batchSize);

                $cost = 0.;

                foreach ($batches as $batch) {
                    $cost += $this->network->roundtrip($batch);
                }

                $cost /= $n;

                $this->steps[] = $cost;

                if (abs($previous - $cost) < $this->minChange) {
                    break 1;
                }

                $previous = $cost;
            }
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the
     * activation of the output neuron.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        if (is_null($this->network)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $activations = $this->network->infer($dataset);

        return array_column($activations, 0);
    }
}

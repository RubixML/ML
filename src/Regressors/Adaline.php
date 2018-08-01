<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Linear;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;

/**
 * Adaline
 *
 * Adaptive Linear Neuron is a type of single layer neural network with a
 * linear output layer. The Adaline converges to the least squares error
 * which is the same error function as linear regression.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Adaline implements Regressor, Online, Persistable
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
     * The minimum change in the weights necessary to continue training.
     *
     * @var float
     */
    protected $minChange;

    /**
     * The underlying computational graph.
     *
     * @var \Rubix\ML\NeuralNet\Network|null
     */
    protected $network;

    /**
     * The sizes of each training step at each epoch.
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
     * @param  float  $minChange
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $epochs = 100, int $batchSize = 10, Optimizer $optimizer = null,
                                float $alpha = 1e-4, float $minChange = 1e-8)
    {
        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Cannot have less than 1 sample'
                . ' per batch.');
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('L2 regularization term must'
                . ' be non-negative.');
        }

        if ($minChange < 0) {
            throw new InvalidArgumentException('Minimum change cannot be less'
                . ' than 0.');
        }

        if (is_null($optimizer)) {
            $optimizer = new Adam();
        }

        $this->epochs = $epochs;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->minChange = $minChange;
    }

    /**
     * Return the sizes of each training step at each epoch.
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
     * @return \Rubix\ML\NeuralNet\Network|null
     */
    public function network() : ?Network
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

        $this->network = new Network(new Input($dataset->numColumns()), [],
            new Linear($this->alpha), $this->optimizer);

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

        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
            . ' continuous features.');
        }

        if (is_null($this->network)) {
            $this->train($dataset);
        }

        $previous = 0.0;

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            $batches = $dataset->randomize()->batch($this->batchSize);

            $step = 0.0;

            foreach ($batches as $batch) {
                $step += $this->network->feed($batch->samples())
                    ->backpropagate($batch->labels());
            }

            $this->steps[] = $step;

            if (abs($previous - $step) < $this->minChange) {
                break 1;
            }

            $previous = $step;
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the
     * activation of the output neuron.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $activations = $this->network->feed($dataset->samples())->activations();

        return array_column($activations, 0);
    }
}

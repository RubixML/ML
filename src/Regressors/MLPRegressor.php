<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\CrossValidation\Metrics\Validation;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use InvalidArgumentException;

class MLPRegressor implements Regressor, Online, Persistable
{
    /**
     * The user-specified hidden layers of the network.
     *
     * @var array
     */
    protected $hidden = [
        //
    ];

    /**
     * The number of training samples to consider per iteholdoutn of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer used to train the network.
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
     * The Validation metric used to validate the performance of the model.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Validation
     */
    protected $metric;

    /**
     * The holdout of training samples to use for validation. i.e. the holdout
     * holdout.
     *
     * @var float
     */
    protected $holdout;

    /**
     * The training window to consider during early stop checking i.e. the last
     * n epochs.
     *
     * @var int
     */
    protected $window;

    /**
     * The amount of validation metric to tolerate when considering early
     * stopping due to max validation score.
     *
     * @var float
     */
    protected $tolerance;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set before the algorithm terminates.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The underlying computational graph.
     *
     * @var \Rubix\ML\NeuralNet\Network|null
     */
    protected $network;

    /**
     * The validation score of each epoch during training.
     *
     * @var array
     */
    protected $progress = [
        //
    ];

    /**
     * @param  array  $hidden
     * @param  int  $batchSize
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer|null  $optimizer
     * @param  float  $alpha
     * @param  \Rubix\ML\CrossValidation\Metrics\Validation|null  $metric
     * @param  float $holdout
     * @param  int  $window
     * @param  float  $tolerance
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $hidden, int $batchSize = 50, Optimizer $optimizer = null,
        float $alpha = 1e-4, Validation $metric = null, float $holdout = 0.2, int $window = 3,
        float $tolerance = 1e-3, int $epochs = PHP_INT_MAX)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException('Cannot have less than 1 sample'
                . ' per batch.');
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Regularization parameter must'
                . ' be non-negative.');
        }

        if ($holdout < 0.01 or $holdout > 1.0) {
            throw new InvalidArgumentException('Holdout ratio must be'
                . ' between 0.01 and 1.');
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Stopping criteria window must'
                . ' be at least 2 epochs.');
        }

        if ($tolerance < -1 or $tolerance > 1) {
            throw new InvalidArgumentException('Validation metric tolerance'
                . ' must be between -1 and 1.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        if (!isset($metric)) {
            $metric = new MeanSquaredError();
        }

        $this->hidden = $hidden;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->metric = $metric;
        $this->holdout = $holdout;
        $this->window = $window;
        $this->tolerance = $tolerance;
        $this->epochs = $epochs;
    }

    /**
     * Return the training progress of the estimator.
     *
     * @return array
     */
    public function progress() : array
    {
        return $this->progress;
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

        $this->network = new Network(new Input($dataset->numColumns()),
            $this->hidden, new Continuous($this->alpha), $this->optimizer);

        $this->network->initialize();

        $this->partial($dataset);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
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

        if (!isset($this->network)) {
            $this->train($dataset);
        }

        list($training, $testing) = $dataset->split(1 - $this->holdout);

        list($min, $max) = $this->metric->range();

        $best = ['score' => -INF, 'snapshot' => null];

        $this->progress = [];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $change = 0.0;

            foreach ($training->randomize()->batch($this->batchSize) as $batch) {
                $change += $this->network->feed($batch->samples())
                    ->backpropagate($batch->labels())
                    ->step();
            }

            $score = $this->metric->score($this, $testing);

            $this->progress[$epoch] = ['score' => $score, 'change' => $change];

            if ($score > $best['score']) {
                $best['score'] = $score;
                $best['snapshot'] = $this->network->read();
            }

            if ($score > ($max - $this->tolerance)) {
                break 1;
            }

            if ($epoch >= $this->window) {
                $window = array_slice($this->progress, -$this->window);

                $worst = $window;
                rsort($worst);

                if ($window === $worst) {
                    break 1;
                }
            }
        }

        if (end($this->progress) !== $best['score']) {
            $this->network->restore($best['snapshot']);
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $results = $this->network->feed($dataset->samples())->activations();

        return array_column($results, 0);
    }
}

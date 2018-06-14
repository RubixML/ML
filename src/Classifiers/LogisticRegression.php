<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Logistic;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;

class LogisticRegression implements Binary, Online, Probabilistic, Persistable
{
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
    protected $threshold;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The underlying computational graph.
     *
     * @param \Rubix\ML\NeuralNet\Network
     */
    protected $network;

    /**
     * @param  int  $batchSize
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @param  float  $alpha
     * @param  float  $threshold
     * @param  int  $epochs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $batchSize = 10, Optimizer $optimizer = null,
                                float $alpha = 1e-4, float $threshold = 1e-4,
                                int $epochs = PHP_INT_MAX)
    {
        if ($batchSize < 1) {
            throw new InvalidArgumentException('Cannot have less than 1 sample'
                . ' per batch.');
        }

        if ($alpha < 0.0) {
            throw new InvalidArgumentException('L2 regularization term must'
                . ' be non-negative.');
        }

        if ($threshold < 0) {
            throw new InvalidArgumentException('Threshold cannot be set to less'
                . ' than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;
        $this->alpha = $alpha;
        $this->threshold = $threshold;
        $this->epochs = $epochs;
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
            new Logistic($dataset->possibleOutcomes(), $this->alpha));

        foreach ($this->network->initialize()->parametric() as $layer) {
            $this->optimizer->initialize($layer);
        }

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

        if (!isset($this->network)) {
            $this->train($dataset);
        }

        $previous = 0.0;

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $change = 0.0;

            foreach ($dataset->randomize()->batch($this->batchSize) as $batch) {
                $this->network->feed($batch->samples())
                    ->backpropagate($batch->labels());

                $step = $this->optimizer->step($this->network->output());

                $this->network->output()->update($step);

                $change += $step->oneNorm();
            }

            if (abs($change - $previous) < $this->threshold) {
                break 1;
            }

            $previous = $change;
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($this->proba($samples) as $probabilities) {
            $best = ['probability' => -INF, 'outcome' => null];

            foreach ($probabilities as $class => $probability) {
                if ($probability > $best['probability']) {
                    $best['probability'] = $probability;
                    $best['outcome'] = $class;
                }
            }

            $predictions[] = $best['outcome'];
        }

        return $predictions;
    }

    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array
    {
        $this->network->feed($samples->samples());

        return $this->network->output()->activations();
    }
}

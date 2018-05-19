<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\NeuralNet\Network;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Persistable;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Binary;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use Rubix\Engine\Estimators\Predictions\Probabalistic;
use InvalidArgumentException;
use RuntimeException;

class Perceptron implements BinaryClassifier, Persistable
{
    /**
     * The output layer of the network.
     *
     * @var \Rubix\Engine\NeuralNet\Layers\Binary
     */
    protected $output;

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
     * @var \Rubix\Engine\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The underlying computational graph.
     *
     * @param \Rubix\Engine\NeuralNet\Network
     */
    protected $network;

    /**
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\Optimizers\Optimizer  $optimizer
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $epochs = 10, int $batchSize = 5,
                                Optimizer $optimizer = null)
    {
        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at'
                . ' least 1 epoch.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than'
                . ' 1.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        $this->batchSize = $batchSize;
        $this->epochs = $epochs;
        $this->optimizer = $optimizer;
    }

    /**
     * Perform mini-batch gradient descent with given optimizer over the training
     * set and update the input weights accordingly.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $this->network = new Network(new Input($dataset->numColumns()),
            [], new Binary($dataset->possibleOutcomes()));

        $this->network->initialize();

        $template = [1 => [array_fill(0, $this->network->input()->width(), 0.0)]];

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            foreach ($this->generateMiniBatches(clone $dataset) as $batch) {
                $accumulated = $template;

                foreach ($batch as $index => $sample) {
                    $this->network->feed($sample);

                    $gradients = $this->network->backpropagate($batch->outcome($index));

                    foreach ($gradients as $i => $layer) {
                        foreach ($layer as $j => $neuron) {
                            foreach ($neuron as $k => $gradient) {
                                $accumulated[$i][$j][$k] += $gradient;
                            }
                        }
                    }

                    $steps = $this->optimizer->step($accumulated);

                    $this->network->output()->update($steps[1]);
                }
            }
        }
    }

    /**
     * Read the activation of the neuron and make a prediction.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
            $activations = $this->network->feed($sample);

            $outcome = current(array_keys($activations));

            $activation = current($activations);

            $predictions[] = new Probabalistic($outcome, $activation);
        }

        return $predictions;
    }

    /**
     * Generate a collection of mini batches from the training data.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return array
     */
    protected function generateMiniBatches(Supervised $dataset) : array
    {
        $batches = [];

        $dataset->randomize();

        while (!$dataset->isEmpty()) {
            $batches[] = $dataset->take($this->batchSize);
        }

        return $batches;
    }
}

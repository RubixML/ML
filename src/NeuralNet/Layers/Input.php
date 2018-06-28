<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

class Input implements Layer
{
    /**
     * The number of input nodes. i.e. feature inputs.
     *
     * @var int
     */
    protected $inputs;

    /**
     * The memoized output activations matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $computed;

    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int
     */
    protected $width;

    /**
     * @param  int  $inputs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of input nodes must'
            . ' be greater than 0.');
        }

        $this->inputs = $inputs;
        $this->width = $inputs + 1;
    }

    /**
     * Return the width of the input layer.
     *
     * @var int
     */
    public function width() : int
    {
        return $this->width;
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function computed() : Matrix
    {
        return $this->computed;
    }

    /**
     * Just return the input vector adding a bias since the input layer does not
     * have any paramters.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($input->getM() !== $this->inputs) {
            throw new InvalidArgumentException('The number of feature columns'
                . ' must equal the number of input inputs. '
                . (string) $input->getM() . ' found, '
                . (string) $this->inputs . ' needed.');
        }

        $biases = MatrixFactory::one(1, $input->getN());

        $this->computed = $input->augmentBelow($biases);

        return $this->computed;
    }
}

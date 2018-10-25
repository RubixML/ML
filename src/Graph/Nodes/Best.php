<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Best
 * 
 * A decision node whose outcome is the most probable class given a set
 * of class labels.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Best extends BinaryNode implements Decision, Leaf
{
    /**
     * The outcome of the decision as the most probable i.e best.
     * 
     * @var string
     */
    protected $outcome;

    /**
     * The probabilities of each discrete outcome.
     * 
     * @var float[]
     */
    protected $probabilities;

    /**
     * The amount of impurity within the node.
     *
     * @var float
     */
    protected $impurity;

    /**
     * The number of labels this node is responsible for.
     *
     * @var int
     */
    protected $n;
    
    /**
     * @param  string  $outcome
     * @param  array  $probabilities
     * @param  float  $impurity
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(string $outcome, array $probabilities, float $impurity, int $n)
    {
        $this->outcome = $outcome;
        $this->probabilities = $probabilities;
        $this->impurity = $impurity;
        $this->n = $n;
    }

    /**
     * Return the outcome of the decision i.e the most probable outcome.
     * 
     * @return string
     */
    public function outcome() : string
    {
        return $this->outcome;
    }

    /**
     * Return the proababilities of each discrete outcome.
     * 
     * @return float[]
     */
    public function probabilities() : array
    {
        return $this->probabilities;
    }

    /**
     * Return the impurity within the node.
     *
     * @return float
     */
    public function impurity() : float
    {
        return $this->impurity;
    }

    /**
     * Return the number of labels within the node.
     *
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }
}

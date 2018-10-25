<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

/**
 * Comparison
 *
 * A decision node that marks a comparison between an input value and the
 * value of the comparison node.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Comparison extends Split implements Decision
{
    /**
     * The amount of impurity that the split introduces.
     *
     * @var float
     */
    protected $impurity;

    /**
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  mixed  $value
     * @param  int  $index
     * @param  array  $groups
     * @param  float  $impurity
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($value, int $index, array $groups, float $impurity)
    {
        parent::__construct($value, $index, $groups);

        if ($impurity < 0.) {
            throw new InvalidArgumentException('Impurity cannot be less than'
                . ' 0.');
        }

        $this->impurity = $impurity;
        $this->n = (int) array_sum(array_map('count', $groups));
    }

    /**
     * Return the impurity score of the node.
     * 
     * @return float
     */
    public function impurity() : float
    {
        return $this->impurity;
    }

    /**
     * Return the number of samples from the training set this node represents.
     * 
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }

    /**
     * Return the decrease in impurity this decision node introduces. A negative
     * impurity means that the decision node actually causes its children to become
     * less pure.
     *
     * @return float
     */
    public function impurityDecrease() : float
    {
        $decrease = $this->impurity;

        if (isset($this->left)) {
            if ($this->left instanceof Decision) {
                $decrease -= $this->left->impurity()
                    * ($this->left->n() / $this->n);
            }
        }

        if (isset($this->right)) {
            if ($this->right instanceof Decision) {
                $decrease -= $this->right->impurity()
                    * ($this->right->n() / $this->n);
            }
        }

        return $decrease;
    }
}

<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildrenTrait;
use Rubix\ML\Exceptions\RuntimeException;

use function array_unique;
use function array_rand;
use function floor;
use function ceil;
use function min;
use function max;
use function getrandmax;
use function rand;

/**
 * Isolator
 *
 * Isolator nodes represent splits in a tree designed to isolate groups into cells by randomly
 * dividing them.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Isolator implements HasBinaryChildren
{
    use HasBinaryChildrenTrait;

    /**
     * The feature column (index) of the split value.
     *
     * @var int
     */
    protected int $column;

    /**
     * The value that the node splits on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right subsets of the training data.
     *
     * @var array{\Rubix\ML\Datasets\Dataset,\Rubix\ML\Datasets\Dataset}
     */
    protected array $subsets;

    /**
     * Factory method to build a isolator node from a dataset using a random split of the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public static function split(Dataset $dataset) : self
    {
        $column = rand(0, $dataset->numFeatures() - 1);

        $values = $dataset->feature($column);

        $type = $dataset->featureType($column);

        if ($type->isContinuous()) {
            $min = min($values);
            $max = max($values);

            $phi = getrandmax() / max(abs($max), abs($min));

            $min = (int) floor($min * $phi);
            $max = (int) ceil($max * $phi);

            $value = rand($min, $max) / $phi;
        } else {
            $offset = array_rand(array_unique($values));

            $value = $values[$offset];
        }

        $subsets = $dataset->splitByFeature($column, $value);

        return new self($column, $value, $subsets);
    }

    /**
     * @param int $column
     * @param string|int|float $value
     * @param array{\Rubix\ML\Datasets\Dataset,\Rubix\ML\Datasets\Dataset} $subsets
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $column, $value, array $subsets)
    {
        $this->column = $column;
        $this->value = $value;
        $this->subsets = $subsets;
    }

    /**
     * Return the feature column (index) of the split value.
     *
     * @return int
     */
    public function column() : int
    {
        return $this->column;
    }

    /**
     * Return the split value.
     *
     * @return int|float|string
     */
    public function value()
    {
        return $this->value;
    }

    /**
     * Return the left and right subsets of the training data.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array{\Rubix\ML\Datasets\Dataset,\Rubix\ML\Datasets\Dataset}
     */
    public function subsets() : array
    {
        if (!isset($this->subsets)) {
            throw new RuntimeException('Subsets property does not exist.');
        }

        return $this->subsets;
    }

    /**
     * Remove any variables carried over from the parent node.
     */
    public function cleanup() : void
    {
        unset($this->subsets);
    }
}

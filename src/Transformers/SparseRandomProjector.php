<?php

namespace Rubix\Engine\Transformers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\Engine\Datasets\Dataset;
use InvalidArgumentException;

class SparseRandomProjector implements Transformer
{
    const ROOT_3 = 1.73205080757;

    const DIRECTIONS = [-1, 0, 0, 0, 0, 1];

    /**
     * The target number of dimensions.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The randomized matrix R.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
    */
    protected $r;

    /**
     * @param  int  $dimensions
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Cannot project onto less than'
                . ' 1 dimension.');
        }

        $this->dimensions = $dimensions;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works'
                . ' with continuous features.');
        }

        $r = [[]];

        for ($i = 0; $i < $dataset->numColumns(); $i++) {
            for ($j = 0; $j < $this->dimensions; $j++) {
                $r[$i][$j] = self::ROOT_3 * self::DIRECTIONS[random_int(0, 5)];
            }
        }

        $this->r = new Matrix($r);
    }

    /**
     * Transform each sample into a dense polynomial feature vector in the degree
     * given.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        $samples = MatrixFactory::create($samples)
            ->multiply($this->r)
            ->getMatrix();
    }
}

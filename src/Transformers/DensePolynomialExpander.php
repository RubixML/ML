<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;
use InvalidArgumentException;

class DensePolynomialExpander implements Transformer
{
    /**
     * The degree of the polynomials to generate. Higher order polynomials are
     * able to fit nonlinear data better, however require extra features to be added
     * to the dataset which grows exponentially.
     *
     * @var int
     */
    protected $degree;

    /**
     * The nmber of columns in the fitted dataset.
     *
     * @var int
     */
    protected $columns;

    /**
     * @param  int  $degree
     * @return void
     */
    public function __construct(int $degree = 2)
    {
        $this->degree = $degree;
    }

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }

        $this->columns = $dataset->numColumns();
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
        foreach ($samples as &$sample) {
            $vector = [];
            $column = 0;

            for ($i = 0; $i < $this->columns; $i++) {
                for ($j = $this->degree; $j > 0; $j--) {
                    $vector[$column] = pow($sample[$i], $j);

                    $column++;
                }
            }

            $sample = $vector;
        }
    }
}

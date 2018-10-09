<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;

interface Online
{
    /**
     * Update the fitting of the transformer.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function update(DataFrame $dataframe) : void;
}

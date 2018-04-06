<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;

interface Preprocessor
{
    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void;

    /**
     * @param  array  $samples
     * @return array
     */
    public function transform(array &$samples) : void;
}

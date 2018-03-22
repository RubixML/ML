<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Math\Matrix;

class CountVectorizer extends Vectorizer
{
    /**
     * Convert a string into a vector where values are token counts.
     *
     * @param  string  $sample
     * @return \Rubix\Engine\Math\Matrix
     */
    public function vectorize(string $sample) : Matrix
    {
        $vector = array_fill_keys($this->vocabulary, 0);

        foreach ($this->tokenizer->tokenize($sample) as $token) {
            if (isset($this->vocabulary[$token])) {
                $vector[$this->vocabulary[$token]] += 1;
            }
        }

        return Matrix::vector($vector);
    }
}

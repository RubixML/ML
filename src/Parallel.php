<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Backend;

interface Parallel
{
    /**
     * Set the parallel processing backend.
     *
     * @param \Rubix\ML\Backends\Backend $backend
     */
    public function setBackend(Backend $backend) : void;
}

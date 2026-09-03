<?php

namespace Rubix\ML\Serializers;

use function Rubix\ML\warn_deprecated;

class RBX extends RBXV1
{
    public function __construct(int $level = 6)
    {
        warn_deprecated('The RBX class is deprecated, use RBXV1 instead.');

        parent::__construct($level);
    }
}

<?php

namespace Rubix\ML\Serializers;

use function Rubix\ML\warn_deprecated;

/**
 * RBX
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @deprecated This serializer is deprecated and will be removed in 4.0, use RBX V1 instead.
 */
class RBX extends RBXV1
{
    /**
     * @param int $level
     */
    public function __construct(int $level = 6)
    {
        warn_deprecated('The RBX serializer is deprecated, use RBX V1 instead.');

        parent::__construct($level);
    }
}

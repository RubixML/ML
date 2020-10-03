<?php

namespace Rubix\ML\Storage;

use Stringable;

/**
 * Datastore.
 *
 * Defines the behaviour of a generic storage repository (filesystem, database etc)
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
interface Datastore extends Reader, Writer, Stringable
{
    //
}

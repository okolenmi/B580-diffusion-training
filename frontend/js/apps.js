/* ---------------------------------------------------------------------------
   apps.js -- entry point of the Apps page (apps.html).

   Two routes: /apps is the library of apps (apps/list_view.js); /apps/<name> is one app (apps/page.js). The pieces
   of the page view live in js/apps/.
   --------------------------------------------------------------------------- */

import { showList } from "./apps/list_view.js";
import { AppPage } from "./apps/page.js";

const url = new URL(location.href);
const route = url.pathname.replace(/\/+$/, "");
const pageName = route.startsWith("/apps/") ? decodeURIComponent(route.slice("/apps/".length)) : null;

if (pageName === null) showList();
else new AppPage(pageName, url.searchParams).show();
